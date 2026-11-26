
"""Evaluate normalized answer agreement with the DeepSeek baseline."""

import json
import re
import unicodedata
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

NO_TARGET = "__no_target__"
NO_READABLE_TEXT = "__no_readable_text__"


def normalize_existence(answer):
    text = str(answer).strip().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())

    if re.match(r"^(yes|yeah|yep)(\b|$)", text):
        return True

    if re.match(r"^(no|nope)(\b|$)", text):
        return False

    if re.search(r"\b(there (is|are)|i (can )?see)\b", text):
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

    no_target_patterns = (
        r"\bthere\s+(?:is|are)\s+no\b",
        r"\bno\s+(?:\w+\s+){0,3}(?:detected|found|visible|present)\b",
        r"\b(?:did not|didn't|could not|couldn't|cannot|can't)\s+"
        r"(?:detect|find|see|identify)\s+(?:any\s+)?"
        r"(?:\w+\s+){0,3}\w+\b",
    )
    if any(re.search(pattern, text) for pattern in no_target_patterns):
        return NO_TARGET

    if re.search(r"\b(unclear|unknown|unsure|cannot determine|can't tell)\b", text):
        return None

    if re.search(r"\b(not|isn't|aren't)\b", text):
        return None

    found = {
        "gray" if color == "grey" else color
        for color in re.findall(r"\b[a-z]+\b", text)
        if color in COLORS
    }

    if not found:
        return None

    return "|".join(sorted(found))


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


def normalize_ocr(answer):
    if answer is None:
        return None

    text = unicodedata.normalize("NFKC", str(answer)).casefold().strip()
    if not text:
        return None

    no_text_patterns = (
        r"\bno\s+(?:clearly\s+)?(?:readable|visible)?\s*text\b",
        r"\bthere\s+is\s+no\s+(?:clearly\s+)?(?:readable|visible)?\s*text\b",
        r"\bno\s+(?:readable\s+)?(?:text|words?)\s+can\s+be\s+read\b",
        r"\bno\s+(?:\w+\s+){0,3}(?:text|writing|words?)\s+"
        r"(?:was|were|is|are)?\s*(?:detected|found|visible|readable)\b",
        r"\b(?:did not|didn't|could not|couldn't|cannot|can't)\s+"
        r"(?:detect|find|read)\s+(?:any\s+)?"
        r"(?:visible\s+)?(?:text|writing|words?)\b",
        r"\bnothing\s+(?:is\s+)?readable\b",
        r"\b(?:cannot|can't|unable\s+to)\s+read\s+(?:any\s+)?(?:visible\s+)?text\b",
        r"\b(?:its|the)\s+(?:text|word|writing)\s+"
        r"(?:cannot|can't)\s+be\s+read\b",
    )
    if any(re.search(pattern, text) for pattern in no_text_patterns):
        return NO_READABLE_TEXT

    quoted_text = re.search(
        r"\*\*(.+?)\*\*|[\"“]([^\"”]+)[\"”]",
        text
    )
    if quoted_text:
        text = quoted_text.group(1) or quoted_text.group(2)
    else:
        described_text = re.search(
            r"\b(?:word|text|label|sign|writing)\b.*?"
            r"\b(?:reads?|says?|is)\s*[:=]?\s*"
            r"(.+?)(?:[,.;!?]|$)",
            text
        )
        if described_text:
            text = described_text.group(1)

    normalized = " ".join(
        re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    )
    return normalized or None


def normalized_edit_similarity(pipeline_text, baseline_text):
    if (pipeline_text == NO_READABLE_TEXT
            or baseline_text == NO_READABLE_TEXT):
        return float(pipeline_text == baseline_text)

    if not pipeline_text or not baseline_text:
        return 0.0

    if len(pipeline_text) > len(baseline_text):
        pipeline_text, baseline_text = baseline_text, pipeline_text

    previous_row = list(range(len(pipeline_text) + 1))
    for baseline_index, baseline_character in enumerate(baseline_text, start=1):
        current_row = [baseline_index]
        for pipeline_index, pipeline_character in enumerate(
            pipeline_text,
            start=1
        ):
            current_row.append(min(
                current_row[-1] + 1,
                previous_row[pipeline_index] + 1,
                previous_row[pipeline_index - 1]
                + (pipeline_character != baseline_character)
            ))
        previous_row = current_row

    edit_distance = previous_row[-1]
    return 1 - edit_distance / max(len(pipeline_text), len(baseline_text))


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

    elif task_type == "ocr":
        pipeline_value = normalize_ocr(task.get("answer", ""))
        baseline_value = normalize_ocr(baseline_result.get("answer", ""))

        if pipeline_value is None or baseline_value is None:
            return None, pipeline_value, baseline_value

        return (
            normalized_edit_similarity(
                pipeline_value,
                baseline_value
            ),
            pipeline_value,
            baseline_value
        )

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
        task_results = pipeline_result.get("task_results", [])
        semantic_type = (
            str(task_results[0].get("semantic_type") or "").lower()
            if len(task_results) == 1 else ""
        )
        is_list = (
            task_type in {"LIST", "LISTING"}
            or semantic_type in {"list", "listing"}
        )
        is_ocr = semantic_type == "ocr" or task_type == "TEXT"
        uses_similarity = is_list or is_ocr

        results.append({
            "image": image,
            "task_type": pipeline_result.get("task_type"),
            "semantic_type": semantic_type,
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
            "accuracy": None if uses_similarity else accuracy,
            "similarity": accuracy if uses_similarity else None,
            "similarity_metric": (
                "jaccard" if is_list
                else "normalized_edit" if is_ocr
                else None
            ),
        })

    evaluated = [
        result for result in results
        if result["accuracy"] is not None
    ]

    correct = sum(result["accuracy"] for result in evaluated)

    # NEW: Calculate batch average similarity for LIST tasks.
    list_results = [
        result for result in results
        if result["similarity_metric"] == "jaccard"
        and result["similarity"] is not None
    ]
    ocr_results = [
        result for result in results
        if result["similarity_metric"] == "normalized_edit"
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
        "ocr_evaluated_images": len(ocr_results),
        "average_ocr_similarity_percent": (
            round(
                sum(result["similarity"] for result in ocr_results)
                / len(ocr_results) * 100,
                2
            )
            if ocr_results else None
        ),
    }

    print("\n========== Cloud Agreement Accuracy ==========")
    print(json.dumps(summary, indent=4, ensure_ascii=False))

    return summary, results
