import json
import tempfile
import unittest
from pathlib import Path

from toolbar.accuracy import (
    NO_READABLE_TEXT,
    evaluate_answer,
    normalize_existence,
    normalize_color,
    normalize_ocr,
    run_accuracy,
)


class ExistenceNormalizationTests(unittest.TestCase):
    def test_detects_absence_after_introductory_text(self):
        answer = (
            "Based on the image provided, there is no zebra. "
            "The image shows a baseball game."
        )

        self.assertIs(normalize_existence(answer), False)

    def test_detects_absence_in_detection_response(self):
        self.assertIs(
            normalize_existence("No zebra was detected by YOLO."),
            False
        )

    def test_detects_presence_after_introductory_text(self):
        self.assertIs(
            normalize_existence(
                "Based on the image, there is a zebra."
            ),
            True
        )

    def test_pipeline_and_baseline_absence_answers_agree(self):
        pipeline_result = {
            "status": "SUCCESS",
            "question": "Is there a zebra?",
            "task_results": [{
                "semantic_type": "existence",
                "answer": "No zebra was detected by YOLO."
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": (
                "Based on the image provided, there is no zebra."
            )
        }

        accuracy, pipeline_value, baseline_value = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(accuracy, 1)
        self.assertIs(pipeline_value, False)
        self.assertIs(baseline_value, False)


class UnavailableAnswerAgreementTests(unittest.TestCase):
    def test_attribute_no_target_answers_agree(self):
        pipeline_result = {
            "status": "SUCCESS",
            "question": "What color is the bus?",
            "task_plan": {"tasks": [{"attribute": "color"}]},
            "task_results": [{
                "semantic_type": "attribute",
                "answer": (
                    "YOLO did not detect any bus, so its color "
                    "cannot be determined."
                )
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": "There is no bus in the image."
        }

        accuracy, pipeline_value, baseline_value = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(accuracy, 1)
        self.assertEqual(pipeline_value, "__no_target__")
        self.assertEqual(baseline_value, "__no_target__")

    def test_attribute_no_target_does_not_match_a_color(self):
        pipeline_result = {
            "status": "SUCCESS",
            "question": "What color is the bus?",
            "task_plan": {"tasks": [{"attribute": "color"}]},
            "task_results": [{
                "semantic_type": "attribute",
                "answer": "YOLO did not detect any bus."
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": "The bus is blue."
        }

        accuracy, _, _ = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(accuracy, 0)

    def test_multiple_colors_are_normalized_as_an_order_independent_set(self):
        self.assertEqual(
            normalize_color("The bus is black and white."),
            "black|white"
        )
        self.assertEqual(
            normalize_color("The bus is white and black."),
            "black|white"
        )

    def test_missing_target_and_multiple_colors_are_counted_as_disagreement(self):
        pipeline_result = {
            "status": "SUCCESS",
            "question": "What color is the bus?",
            "task_plan": {"tasks": [{"attribute": "color"}]},
            "task_results": [{
                "semantic_type": "attribute",
                "answer": (
                    "YOLO did not detect any bus, so its color "
                    "cannot be determined."
                )
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": (
                "Based on the image, the bus is black and white."
            )
        }

        accuracy, pipeline_value, baseline_value = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(accuracy, 0)
        self.assertEqual(pipeline_value, "__no_target__")
        self.assertEqual(baseline_value, "black|white")

    def test_ocr_no_text_answers_normalize_to_same_value(self):
        self.assertEqual(
            normalize_ocr("YOLO did not detect any text."),
            NO_READABLE_TEXT
        )
        self.assertEqual(
            normalize_ocr("No refrigerator is visible."),
            NO_READABLE_TEXT
        )
        self.assertEqual(
            evaluate_answer(
                {
                    "status": "SUCCESS",
                    "task_results": [{
                        "semantic_type": "ocr",
                        "answer": (
                            "YOLO did not detect any refrigerator, "
                            "so its text cannot be read."
                        )
                    }]
                },
                {
                    "status": "SUCCESS",
                    "answer": "No refrigerator is visible."
                }
            )[0],
            1
        )
        self.assertEqual(
            normalize_ocr(
                "YOLO did not detect any refrigerator, "
                "so its text cannot be read."
            ),
            NO_READABLE_TEXT
        )
        self.assertEqual(
            normalize_ocr("There is no readable text in the image."),
            NO_READABLE_TEXT
        )
        self.assertIsNone(
            normalize_ocr(
                "Unable to determine whether the refrigerator is visible; "
                "OCR was skipped."
            )
        )

        pipeline_result = {
            "status": "SUCCESS",
            "task_results": [{
                "semantic_type": "ocr",
                "answer": "YOLO did not detect any text."
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": "There is no readable text in the image."
        }

        similarity, _, _ = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(similarity, 1.0)

    def test_ocr_answer_matches_text_embedded_in_explanation(self):
        pipeline_result = {
            "status": "SUCCESS",
            "task_results": [{
                "semantic_type": "ocr",
                "answer": "KUHL"
            }]
        }
        baseline_result = {
            "status": "SUCCESS",
            "answer": (
                "Based on the image, the word written on the fridge "
                "is **KUHL**."
            )
        }

        similarity, pipeline_value, baseline_value = evaluate_answer(
            pipeline_result,
            baseline_result
        )

        self.assertEqual(similarity, 1.0)
        self.assertEqual(pipeline_value, "kuhl")
        self.assertEqual(baseline_value, "kuhl")

    def test_explicitly_read_text_is_not_misclassified_as_no_detection(self):
        answer = (
            "There is no fridge visible, but the sign says **EXIT**."
        )

        self.assertEqual(normalize_ocr(answer), "exit")

    def test_ocr_no_detection_uses_exact_agreement_and_detected_text_similarity(self):
        pipeline_results = [
            {
                "image": "no_text_match.jpg",
                "task_type": "ATTRIBUTE",
                "question": (
                    "If a refrigerator is visible, transcribe the text "
                    "written on it."
                ),
                "status": "SUCCESS",
                "task_plan": {
                    "tasks": [{
                        "task_type": "attribute",
                        "target": "refrigerator",
                        "attribute": "visible"
                    }]
                },
                "task_results": [{
                    "semantic_type": "attribute",
                    "answer": (
                        "YOLO did not detect any refrigerator, "
                        "so its text cannot be read."
                    )
                }]
            },
            {
                "image": "no_text_mismatch.jpg",
                "task_type": "ATTRIBUTE",
                "question": (
                    "If a refrigerator is visible, transcribe the text "
                    "written on it."
                ),
                "status": "SUCCESS",
                "task_plan": {
                    "tasks": [{
                        "task_type": "attribute",
                        "target": "refrigerator",
                        "attribute": "visible"
                    }]
                },
                "task_results": [{
                    "semantic_type": "attribute",
                    "answer": (
                        "YOLO did not detect any refrigerator, "
                        "so its text cannot be read."
                    )
                }]
            },
            {
                "image": "read_text.jpg",
                "task_type": "TEXT",
                "status": "SUCCESS",
                "task_results": [{
                    "semantic_type": "ocr",
                    "answer": "KUHL"
                }]
            }
        ]
        baseline_results = [
            {
                "image": "no_text_match.jpg",
                "status": "SUCCESS",
                "answer": "There is no readable text in the image."
            },
            {
                "image": "no_text_mismatch.jpg",
                "status": "SUCCESS",
                "answer": "The word on the fridge is **KUHL**."
            },
            {
                "image": "read_text.jpg",
                "status": "SUCCESS",
                "answer": "Based on the image, the word on the fridge is **KUHL**."
            }
        ]

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir)
            (output_path / "results.json").write_text(
                json.dumps(pipeline_results),
                encoding="utf-8"
            )
            (output_path / "baseline_results.json").write_text(
                json.dumps(baseline_results),
                encoding="utf-8"
            )

            summary, results = run_accuracy(output_path)

        self.assertEqual(summary["evaluated_images"], 3)
        self.assertEqual(summary["agreement_accuracy_percent"], 66.67)
        self.assertEqual(summary["correct_images"], 2)
        self.assertEqual(summary["ocr_evaluated_images"], 1)
        self.assertEqual(
            results[0]["accuracy"],
            1
        )
        self.assertIsNone(results[0]["similarity"])
        self.assertEqual(results[1]["accuracy"], 0)
        self.assertIsNone(results[1]["similarity"])
        self.assertIsNone(results[2]["accuracy"])
        self.assertEqual(results[2]["similarity"], 1.0)
        self.assertEqual(results[0]["agreement"], 1)
        self.assertEqual(results[1]["agreement"], 0)
        self.assertEqual(results[2]["agreement"], 1)


if __name__ == "__main__":
    unittest.main()
