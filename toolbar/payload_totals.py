
"""
Per-image composite payload measurement.

Measurement unit:
    One image + one question + one routing decision.

Total payload:
    Static model payload
    + Runtime data payload
    + Network communication payload

This metric represents logical resource data volume,
not physical network traffic or peak memory usage.
"""

import json
from pathlib import Path
from toolbar import config


# ==============================
# Runtime Payload Configuration
# ==============================

LOCAL_FIELDS = {
    "qwen_classification": (
        "input_payload_bytes",
        "output_payload_bytes",
        "parsed_output_payload_bytes",
    ),

    "router_decision": (
        "input_payload_bytes",
        "output_payload_bytes",
    ),

    "yolo_detection": (
        "input_payload_bytes",
        "input_tensor_bytes",
        "raw_output_tensor_bytes",
        "postprocessed_boxes_tensor_bytes",
        "output_payload_bytes",
    ),

    "local_detection_answer": (
        "input_payload_bytes",
        "output_payload_bytes",
    ),

    "smolvlm_inference": (
        "local_api_request_bytes",
        "local_api_response_bytes",
    ),
}

MODEL_KEYS = {
    "qwen_classification": "Qwen",
    "smolvlm_inference": "SmolVLM",
}

PAYLOAD_FIELDS = (
    "static_payload_bytes",
    "runtime_payload_bytes",
    "network_payload_bytes",
    "total_payload_bytes",
)


# ==============================
# Numeric Utilities
# ==============================

def valid_bytes(value):
    return type(value) is int and value >= 0


def sum_known(values):
    values = list(values)

    if not all(valid_bytes(value) for value in values):
        return None

    return sum(values)


def average_known(values):
    values = list(values)

    if not values:
        return None

    total = sum_known(values)

    if total is None:
        return None

    return round(total / len(values), 2)


# ==============================
# Model Identification
# ==============================

def model_key(stage):
    stage_name = stage.get("stage")

    if stage_name == "yolo_detection":

        model_name = Path(
            str(stage.get("model", ""))
        ).stem

        model_name = model_name.removesuffix(
            "_ncnn_model"
        )

        return model_name.replace(
            "yolo", "YOLO", 1
        )

    return MODEL_KEYS.get(stage_name)


# ==============================
# Static Model Payload
# ==============================

def deployed_model_size(key, stage, profiles):

    if key == "Qwen":

        path = Path(
            stage.get("model_path")
            or (
                config.ROOT
                / "models"
                / "Qwen"
                / "Qwen3-0.6B-Q4_0.gguf"
            )
        )

        files = [path]

    elif stage.get("stage") == "yolo_detection":

        path = stage.get("model_path")

        if not path and model_key({
            "stage": "yolo_detection",
            "model": config.YOLO_MODEL,
        }) != key:
            return None

        path = Path(path or config.YOLO_MODEL)

        if path.is_dir():

            files = [
                path / "model.ncnn.bin",
                path / "model.ncnn.param",
            ]

            if (path / "metadata.yaml").is_file():
                files.append(path / "metadata.yaml")

        else:

            files = (
                [path]
                if path.suffix.lower() in {".pt", ".onnx"}
                else []
            )

    else:

        return (profiles.get(key) or {}).get(
            "total_model_size_bytes"
        )

    try:

        if files and all(path.is_file() for path in files):
            return sum(
                path.stat().st_size
                for path in files
            )

    except OSError:
        pass

    return None


def measure_static_payload(stages, profiles):

    used_models = {}

    for stage in stages:

        stage_name = stage.get("stage")

        if stage_name == "deepseek_cloud_inference":
            continue

        key = model_key(stage)

        if key is None:
            continue

        # Count each model once within this request.
        # Do not deduplicate across different images.
        if key not in used_models:

            used_models[key] = deployed_model_size(
                key,
                stage,
                profiles,
            )

    static_bytes = sum_known(
        used_models.values()
    )

    return static_bytes, used_models


# ==============================
# Runtime Data Payload
# ==============================

def measure_runtime_payload(stages):

    runtime_stages = []

    for stage in stages:

        stage_name = stage.get("stage")

        # Cloud communication is measured separately.
        if stage_name == "deepseek_cloud_inference":
            continue

        fields = LOCAL_FIELDS.get(stage_name)

        if fields is None:
            continue

        components = {
            field: stage.get(field)
            for field in fields
        }

        stage_bytes = sum_known(
            components.values()
        )

        runtime_stages.append({
            "stage": stage_name,
            "model": stage.get("model"),
            "payload_bytes": stage_bytes,
            "components": components,
        })

    runtime_bytes = sum_known(
        stage["payload_bytes"]
        for stage in runtime_stages
    )

    return runtime_bytes, runtime_stages


# ==============================
# Network Communication Payload
# ==============================

def measure_network_payload(result):

    communication = result.get(
        "communication_metrics"
    ) or {}

    metrics = result.get("metrics") or {}

    request_bytes = metrics.get(
        "edge_cloud_request_bytes",
        result.get(
            "edge_cloud_request_bytes",
            communication.get("request_bytes"),
        ),
    )

    response_bytes = metrics.get(
        "edge_cloud_response_bytes",
        result.get(
            "edge_cloud_response_bytes",
            communication.get("response_bytes"),
        ),
    )

    tasks = result.get("task_results") or []

    if (
        result.get("status") == "SUCCESS"
        and tasks
        and metrics.get("edge_cloud_total_bytes") == 0
        and metrics.get("edge_to_cloud_bytes") == 0
        and all(
            task.get("route") in {"DETECTIONS", "FULL_IMAGE"}
            for task in tasks
        )
    ):

        request_bytes = 0
        response_bytes = 0

    total_bytes = sum_known([
        request_bytes,
        response_bytes,
    ])

    return total_bytes, {
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "total_bytes": total_bytes,
        "estimated_request_bytes": metrics.get(
            "edge_to_cloud_bytes",
            result.get("request_bytes"),
        ),
    }


# ==============================
# Per-image Total Payload
# ==============================

def measure(result, profiles, label="pipeline"):

    stages = result.get("runtime_metrics")

    missing_fields = []

    if (
        label == "baseline"
        and result.get("status") == "SUCCESS"
    ):

        static_bytes = 0
        used_models = {}

        runtime_bytes = 0
        runtime_stages = []

    elif isinstance(stages, list) and stages:

        static_bytes, used_models = measure_static_payload(
            stages,
            profiles,
        )

        runtime_bytes, runtime_stages = measure_runtime_payload(
            stages
        )

        tasks = result.get("task_results") or []

        expected = {
            "qwen_classification": 1,

            "router_decision": len(tasks),

            "yolo_detection": int(
                any(
                    task.get(
                        "requested_route",
                        task.get("route"),
                    ) in {"DETECTIONS", "CROP"}
                    for task in tasks
                )
            ),

            "local_detection_answer": sum(
                task.get("route") == "DETECTIONS"
                for task in tasks
            ),

            "smolvlm_inference": sum(
                task.get("route") == "FULL_IMAGE"
                for task in tasks
            ),
        }

        if not tasks:
            missing_fields.append("task_results")

        for stage_name, count in expected.items():

            actual_count = sum(
                stage.get("stage") == stage_name
                for stage in stages
            )

            if actual_count != count:

                missing_fields.append(
                    f"runtime_metrics.{stage_name} "
                    f"(expected {count} stages)"
                )

        if missing_fields:

            static_bytes = None
            runtime_bytes = None

    else:

        static_bytes = None
        used_models = {}

        runtime_bytes = None
        runtime_stages = []

        missing_fields.append("runtime_metrics")

    network_bytes, network_components = measure_network_payload(
        result
    )

    missing_fields.extend(
        f"static_models.{key}"
        for key, size in used_models.items()
        if not valid_bytes(size)
    )

    for stage in runtime_stages:

        missing_fields.extend(
            f"runtime_metrics.{stage['stage']}.{field}"
            for field, size in stage["components"].items()
            if not valid_bytes(size)
        )

    missing_fields.extend(
        f"network_components.{field}"
        for field in ("request_bytes", "response_bytes")
        if not valid_bytes(network_components[field])
    )

    total_bytes = sum_known([
        static_bytes,
        runtime_bytes,
        network_bytes,
    ])

    return {
        "static_models": used_models,

        "runtime_stages": runtime_stages,

        "network_components": network_components,

        "static_payload_bytes": static_bytes,

        "runtime_payload_bytes": runtime_bytes,

        "network_payload_bytes": network_bytes,

        "total_payload_bytes": total_bytes,

        "missing_fields": missing_fields,

        "measurement_complete": (
            not missing_fields
            and total_bytes is not None
            and result.get("status") == "SUCCESS"
        ),
    }


# ==============================
# Per-image Summary
# ==============================

def summarize(rows):

    complete_rows = [
        row for row in rows
        if row.get("measurement_complete")
    ]

    totals = [
        row["total_payload_bytes"]
        for row in complete_rows
    ]

    summary = {
        "total_images": len(rows),

        "evaluated_images": len(complete_rows),

        "not_evaluated_images": (
            len(rows) - len(complete_rows)
        ),

        "average_static_payload_bytes": (
            average_known(
                row["static_payload_bytes"]
                for row in complete_rows
            )
        ),

        "average_runtime_payload_bytes": (
            average_known(
                row["runtime_payload_bytes"]
                for row in complete_rows
            )
        ),

        "average_network_payload_bytes": (
            average_known(
                row["network_payload_bytes"]
                for row in complete_rows
            )
        ),

        "average_total_payload_bytes": (
            average_known(totals)
        ),

        "min_total_payload_bytes": (
            min(totals) if totals else None
        ),

        "max_total_payload_bytes": (
            max(totals) if totals else None
        ),
    }

    return summary


# ==============================
# JSON Utilities
# ==============================

def read_json(path):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8-sig"
        )
    )


def write_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
        ) + "\n",
        encoding="utf-8",
    )


# ==============================
# Latency Utilities
# ==============================

def get_latency(result, label):

    if label == "pipeline":

        return result.get(
            "metrics", {}
        ).get("total_latency_ms")

    return result.get(
        "end_to_end_latency_ms"
    )


def average_latency(results, label):

    values = [
        get_latency(result, label)
        for result in results
    ]

    if not values:
        return None

    if any(
        type(value) not in (int, float)
        or value < 0
        for value in values
    ):
        return None

    return round(
        sum(values) / len(values),
        3,
    )


# ==============================
# Result Integration
# ==============================

def update_results(
    output_dir,
    profiles,
    label,
    results_name,
    summary_name,
):

    results_path = output_dir / results_name
    summary_path = output_dir / summary_name

    results = read_json(results_path)

    for result in results:

        payload = measure(
            result,
            profiles,
            label,
        )

        result["payload_metrics"] = payload

        # Expose the primary metric directly.
        result["total_payload_bytes"] = (
            payload["total_payload_bytes"]
        )

    payload_summary = summarize([
        result["payload_metrics"]
        for result in results
    ])

    summary = read_json(summary_path)

    summary["payload_metrics"] = (
        payload_summary
    )

    summary["average_total_payload_bytes"] = (
        payload_summary[
            "average_total_payload_bytes"
        ]
    )

    write_json(results_path, results)
    write_json(summary_path, summary)

    return results


# ==============================
# Per-image Paired Comparison
# ==============================

def compare_images(
    pipeline_results,
    baseline_results,
    accuracy_results,
):

    baseline_index = {
        (result["image"], result["question"]): result
        for result in baseline_results
    }

    accuracy_index = {
        result["image"]: result
        for result in accuracy_results
    }

    comparisons = []

    for pipeline_result in pipeline_results:

        key = (
            pipeline_result["image"],
            pipeline_result["question"],
        )

        baseline_result = baseline_index.get(key)

        pipeline_payload = pipeline_result[
            "payload_metrics"
        ]

        baseline_payload = (
            baseline_result["payload_metrics"]
            if baseline_result else {}
        )

        diagnostics = []

        for side, result, payload in (
            ("pipeline", pipeline_result, pipeline_payload),
            ("baseline", baseline_result, baseline_payload),
        ):

            if (
                result is None
                or not payload.get("measurement_complete")
            ):

                if result is None:

                    reason = "missing image/question match"

                elif result.get("status") != "SUCCESS":

                    reason = "run failed"

                else:

                    reason = "payload measurement incomplete"

                diagnostics.append({
                    "side": side,

                    "reason": reason,

                    "missing_fields": (
                        list(payload.get("missing_fields", []))
                        + [
                            field
                            for field in PAYLOAD_FIELDS
                            if payload.get(field) is None
                        ]
                    ),
                })

        accuracy_result = accuracy_index.get(
            pipeline_result["image"], {}
        )

        pipeline_total = pipeline_payload[
            "total_payload_bytes"
        ]

        baseline_total = baseline_payload.get(
            "total_payload_bytes"
        )

        complete = (
            pipeline_payload["measurement_complete"]
            and baseline_payload.get(
                "measurement_complete", False
            )
        )

        payload_difference = None
        payload_reduction_percent = None

        if complete:

            payload_difference = (
                baseline_total - pipeline_total
            )

            if baseline_total > 0:

                payload_reduction_percent = round(
                    payload_difference
                    / baseline_total
                    * 100,
                    3,
                )

        comparisons.append({
            "image": pipeline_result["image"],

            "question": pipeline_result["question"],

            "task_type": pipeline_result.get(
                "task_type"
            ),

            "route": pipeline_result.get(
                "route"
            ),

            "pipeline": {
                "status": pipeline_result.get(
                    "status"
                ),

                "payload": pipeline_payload,

                "total_latency_ms": get_latency(
                    pipeline_result,
                    "pipeline",
                ),
            },

            "baseline": {
                "status": (baseline_result or {}).get(
                    "status"
                ),

                "payload": baseline_payload,

                "total_latency_ms": get_latency(
                    baseline_result or {},
                    "baseline",
                ),
            },

            "accuracy": accuracy_result.get(
                "accuracy"
            ),

            "similarity": accuracy_result.get(
                "similarity"
            ),

            "payload_difference_bytes": (
                payload_difference
            ),

            "payload_reduction_percent": (
                payload_reduction_percent
            ),

            "measurement_complete": complete,

            "diagnostics": diagnostics,
        })

    return comparisons


# ==============================
# Final Evaluation
# ==============================

def finalize_payload(
    output_dir,
    profiles,
    accuracy,
):

    output_dir = Path(output_dir)

    profiles = (
        profiles.get("models", profiles)
        if profiles else {}
    )

    pipeline_results = update_results(
        output_dir=output_dir,
        profiles=profiles,
        label="pipeline",
        results_name="results.json",
        summary_name="summary.json",
    )

    baseline_results = update_results(
        output_dir=output_dir,
        profiles=profiles,
        label="baseline",
        results_name="baseline_results.json",
        summary_name="baseline_summary.json",
    )

    accuracy_results = read_json(
        output_dir / "accuracy_results.json"
    )

    comparisons = compare_images(
        pipeline_results,
        baseline_results,
        accuracy_results,
    )

    valid_pairs = [
        result for result in comparisons
        if result["measurement_complete"]
    ]

    pipeline_summary = summarize([
        result["pipeline"]["payload"]
        for result in valid_pairs
    ])

    baseline_summary = summarize([
        result["baseline"]["payload"]
        for result in valid_pairs
    ])

    pipeline_average = pipeline_summary[
        "average_total_payload_bytes"
    ]

    baseline_average = baseline_summary[
        "average_total_payload_bytes"
    ]

    reduction_percent = None

    if (
        pipeline_average is not None
        and baseline_average is not None
        and baseline_average > 0
    ):

        reduction_percent = round(
            (
                baseline_average
                - pipeline_average
            )
            / baseline_average
            * 100,
            3,
        )

    successful_pairs = [
        result for result in comparisons
        if (
            result["pipeline"]["status"] == "SUCCESS"
            and result["baseline"]["status"] == "SUCCESS"
        )
    ]

    valid_keys = {
        (result["image"], result["question"])
        for result in successful_pairs
    }

    paired_pipeline = [
        result for result in pipeline_results
        if (
            result["image"],
            result["question"],
        ) in valid_keys
    ]

    paired_baseline = [
        result for result in baseline_results
        if (
            result["image"],
            result["question"],
        ) in valid_keys
    ]

    pipeline_summary["latency_evaluated_images"] = len(
        paired_pipeline
    )

    baseline_summary["latency_evaluated_images"] = len(
        paired_baseline
    )

    pipeline_summary["average_latency_ms"] = (
        average_latency(
            paired_pipeline,
            "pipeline",
        )
    )

    baseline_summary["average_latency_ms"] = (
        average_latency(
            paired_baseline,
            "baseline",
        )
    )

    paired_accuracy = [
        result["accuracy"]
        for result in successful_pairs
        if result["accuracy"] is not None
    ]

    paired_similarity = [
        result["similarity"]
        for result in successful_pairs
        if result["similarity"] is not None
    ]

    summary = {
        "measurement_unit": (
            "one_image_one_question_one_request"
        ),

        "payload_definition": (
            "static_model_payload + "
            "runtime_data_payload + "
            "network_communication_payload"
        ),

        "static_accounting": (
            "count each used model once per request; "
            "no deduplication across images"
        ),

        "payload_scope": (
            "edge model storage, selected observable "
            "runtime representations, and cloud body bytes"
        ),

        "total_paired_images": sum(
            result["baseline"]["status"] is not None
            for result in comparisons
        ),

        "unmatched_pipeline_images": sum(
            result["baseline"]["status"] is None
            for result in comparisons
        ),

        "evaluated_paired_images": len(valid_pairs),

        "not_evaluated_paired_images": (
            sum(
                result["baseline"]["status"] is not None
                for result in comparisons
            ) - len(valid_pairs)
        ),

        "pipeline": pipeline_summary,

        "baseline": baseline_summary,

        "average_payload_reduction_percent": (
            reduction_percent
        ),

        "paired_accuracy": {
            "evaluated_images": len(
                paired_accuracy
            ),

            "agreement_accuracy_percent": (
                round(
                    sum(paired_accuracy)
                    / len(paired_accuracy)
                    * 100,
                    2,
                )
                if paired_accuracy else None
            ),

            "average_similarity_percent": (
                round(
                    sum(paired_similarity)
                    / len(paired_similarity)
                    * 100,
                    2,
                )
                if paired_similarity else None
            ),
        },

        "overall_accuracy": accuracy,

        "payload_diagnostics": [
            {
                "image": result["image"],
                "question": result["question"],
                **diagnostic,
            }
            for result in comparisons
            for diagnostic in result["diagnostics"]
        ],
    }

    write_json(
        output_dir / "evaluation_results.json",
        comparisons,
    )

    write_json(
        output_dir / "evaluation_summary.json",
        summary,
    )

    print(
        "\n========== Final Evaluation =========="
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=4,
        )
    )

    return summary
