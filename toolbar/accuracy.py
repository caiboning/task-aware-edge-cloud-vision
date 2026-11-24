
"""Evaluate normalized answer agreement with the DeepSeek baseline."""

import json
import re
from collections import Counter
from pathlib import Path

from toolbar.router import COCO_CLASSES, CLASS_ALIASES


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}


COLORS = {
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown", "black", "white", "gray", "grey", "silver",
    "gold", "beige", "tan", "maroon", "navy", "cyan", "teal",
}


def normalize_existence(answer):
    text = str(answer).strip().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())

    if re.match(r"^(yes|yeah|yep)(\b|$)", text):
        return True

    if re.match(r"^(no|nope)(\b|$)", text):
        return False

    if re.match(r"^(there (is|are)|i (can )?see)\b", text):
        if re.search(r"\b(no|not|none|zero)\b", text):
            return False
        return True

    return None


def normalize_counting(answer):
    text = str(answer).strip().lower().replace(",", "")

    if re.search(r"\b(no|none|zero)\b", text) and not re.search(r"\b\d+\b", text):
        return 0

    digits = re.findall(r"\b\d+\b", text)
    if digits:
        return int(digits[0])

    words = re.findall(
        r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b",
        text
    )

    return NUMBER_WORDS[words[0]] if words else None


def normalize_color(answer):
    text = str(answer).strip().lower()

    if re.search(r"\b(unclear|unknown|unsure|cannot determine|can't tell)\b", text):
        return None

    if re.search(r"\b(not|isn't|aren't)\b", text):
        return None

    found = {
        "gray" if color == "grey" else color
        for color in re.findall(r"\b[a-z]+\b", text)
        if color in COLORS
    }

    return next(iter(found)) if len(found) == 1 else None


# NEW: Normalize LIST answers to COCO object class sets.
def normalize_list(answer):
    if answer is None:
        return None

    text = str(answer).strip().lower()
    if not text:
        return None

    found = set()

    for class_name in COCO_CLASSES:
        aliases = [class_name] + CLASS_ALIASES.get(class_name, [])

        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

            if re.search(pattern, text):
                found.add(class_name)
                break

    if not found:
        if re.search(
            r"\b(no objects?|nothing|none|empty)\b", text
        ):
            return set()
        return None

    return found


# NEW: Calculate Jaccard similarity for LIST answers.
def list_similarity(pipeline_value, baseline_value):
    union = pipeline_value | baseline_value

    if not union:
        return 1.0

    return len(pipeline_value & baseline_value) / len(union)


def evaluate_answer(pipeline_result, baseline_result):
    if (pipeline_result.get("status") != "SUCCESS"
            or not baseline_result
            or baseline_result.get("status") != "SUCCESS"):
        return None, None, None

    tasks = pipeline_result.get("task_results", [])
    if len(tasks) != 1:
        return None, None, None

    task = tasks[0]
    task_type = str(
        task.get("semantic_type") or task.get("task_type") or ""
    ).lower()

    question = str(pipeline_result.get("question") or "")

    if task_type == "existence":
        normalizer = normalize_existence

    elif task_type == "counting":
        normalizer = normalize_counting

    elif task_type == "attribute":
        plan_tasks = pipeline_result.get("task_plan", {}).get("tasks", [])

        attribute = (
            str(plan_tasks[0].get("attribute") or "").lower()
            if len(plan_tasks) == 1 else ""
        )

        if attribute not in {"color", "colour"} and not re.search(
            r"\bcolou?rs?\b", question, re.I
        ):
            return None, None, None

        normalizer = normalize_color

    # NEW: LIST is evaluated separately using Jaccard similarity.
    elif task_type in {"listing", "list"}:
        pipeline_value = normalize_list(task.get("answer", ""))
        baseline_value = normalize_list(baseline_result.get("answer", ""))

        if pipeline_value is None or baseline_value is None:
            return None, pipeline_value, baseline_value

        return (
            list_similarity(pipeline_value, baseline_value),
            pipeline_value,
            baseline_value
        )

    else:
        return None, None, None

    pipeline_value = normalizer(task.get("answer", ""))
    baseline_value = normalizer(baseline_result.get("answer", ""))

    if pipeline_value is None or baseline_value is None:
        return None, pipeline_value, baseline_value

    return int(pipeline_value == baseline_value), pipeline_value, baseline_value


def run_accuracy(output_dir):
    """Compare paired JSON results after both runs have completed."""

    output_dir = Path(output_dir)

    pipeline_results = json.loads(
        (output_dir / "results.json").read_text(encoding="utf-8")
    )

    baseline_results = json.loads(
        (output_dir / "baseline_results.json").read_text(encoding="utf-8")
    )

    baseline_by_image = {
        result["image"]: result
        for result in baseline_results
    }

    results = []

    for pipeline_result in pipeline_results:
        image = pipeline_result["image"]
        baseline_result = baseline_by_image.get(image)

        accuracy, pipeline_value, baseline_value = evaluate_answer(
            pipeline_result, baseline_result
        )

        # NEW: Distinguish LIST similarity from exact-match accuracy.
        task_type = str(pipeline_result.get("task_type") or "").upper()
        is_list = task_type in {"LIST", "LISTING"}

        results.append({
            "image": image,
            "task_type": pipeline_result.get("task_type"),
            "pipeline_answer": pipeline_result.get("answer"),
            "baseline_answer": (
                baseline_result.get("answer")
                if baseline_result else None
            ),
            "pipeline_normalized": (
                sorted(pipeline_value)
                if is_list and isinstance(pipeline_value, set)
                else pipeline_value
            ),
            "baseline_normalized": (
                sorted(baseline_value)
                if is_list and isinstance(baseline_value, set)
                else baseline_value
            ),
            "accuracy": None if is_list else accuracy,
            "similarity": accuracy if is_list else None,
        })

    evaluated = [
        result for result in results
        if result["accuracy"] is not None
    ]

    correct = sum(result["accuracy"] for result in evaluated)

    # NEW: Calculate batch average similarity for LIST tasks.
    list_results = [
        result for result in results
        if result["task_type"] in {"LIST", "LISTING"}
        and result["similarity"] is not None
    ]

    summary = {
        "total_images": len(results),
        "evaluated_images": len(evaluated),
        "not_evaluated_images": len(results) - len(evaluated),
        "correct_images": correct,
        "incorrect_images": len(evaluated) - correct,
        "agreement_accuracy_percent": (
            round(correct / len(evaluated) * 100, 2)
            if evaluated else None
        ),
        "task_counts": dict(
            Counter(result["task_type"] for result in evaluated)
        ),
        # NEW: LIST similarity summary.
        "list_evaluated_images": len(list_results),
        "average_similarity_percent": (
            round(
                sum(result["similarity"] for result in list_results)
                / len(list_results) * 100,
                2
            )
            if list_results else None
        ),
    }

    (output_dir / "accuracy_results.json").write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    (output_dir / "accuracy_summary.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== Cloud Agreement Accuracy ==========")
    print(json.dumps(summary, indent=4, ensure_ascii=False))

    return summary
