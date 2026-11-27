import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import baseline
from toolbar import config


class BaselineTargetedQuestionFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.image_dir = self.root / "images"
        self.output_dir = self.root / "output"
        self.image_dir.mkdir()
        self.output_dir.mkdir()
        self.image_path = self.image_dir / "sample.jpg"
        self.image_path.write_bytes(b"test image")

        self.image_dir_patch = patch.object(
            config,
            "IMAGE_DIR",
            self.image_dir
        )
        self.output_dir_patch = patch.object(
            config,
            "PIPELINE_OUTPUT_DIR",
            self.output_dir
        )
        self.image_dir_patch.start()
        self.output_dir_patch.start()

    def tearDown(self):
        self.output_dir_patch.stop()
        self.image_dir_patch.stop()
        self.temp_dir.cleanup()

    def pipeline_results(self):
        return [{
            "image": "sample.jpg",
            "task_plan": {
                "tasks": [{
                    "task_type": "attribute",
                    "target": "refrigerator",
                    "attribute": "visible"
                }]
            },
            "task_results": [{
                "semantic_type": "attribute",
                "target_class": "refrigerator"
            }]
        }]

    def load_baseline_results(self):
        return json.loads(
            (self.output_dir / "baseline_results.json").read_text(
                encoding="utf-8"
            )
        )

    def test_skips_ocr_call_when_target_is_absent(self):
        with (
            patch(
                "baseline.ask_deepseek",
                return_value=("NO", 0.1, 100)
            ) as ask_deepseek,
            patch(
                "baseline.cloud_body_sizes",
                return_value=(120, 40)
            ) as body_sizes
        ):
            baseline.run_baseline(
                "What text is on the refrigerator?",
                image_paths=[self.image_path],
                pipeline_results=self.pipeline_results()
            )

        self.assertEqual(ask_deepseek.call_count, 1)
        self.assertEqual(body_sizes.call_args.args[1], 1)
        result = self.load_baseline_results()[0]
        self.assertEqual(result["answer"], "No refrigerator is visible.")
        self.assertEqual(result["existence_check_answer"], "NO")
        self.assertIsNone(result["followup_answer"])
        self.assertEqual(result["request_bytes"], 100)
        self.assertEqual(result["edge_cloud_total_bytes"], 160)

    def test_asks_for_text_only_after_target_is_confirmed_present(self):
        with (
            patch(
                "baseline.ask_deepseek",
                side_effect=[
                    ("YES", 0.1, 100),
                    ("KUHL", 0.2, 150)
                ]
            ) as ask_deepseek,
            patch(
                "baseline.cloud_body_sizes",
                return_value=(300, 80)
            ) as body_sizes
        ):
            baseline.run_baseline(
                "What text is on the refrigerator?",
                image_paths=[self.image_path],
                pipeline_results=self.pipeline_results()
            )

        self.assertEqual(ask_deepseek.call_count, 2)
        self.assertEqual(body_sizes.call_args.args[1], 2)
        second_question = ask_deepseek.call_args_list[1].args[1]
        self.assertIn("refrigerator", second_question)
        self.assertIn("Read only text", second_question)
        result = self.load_baseline_results()[0]
        self.assertEqual(result["answer"], "KUHL")
        self.assertEqual(result["existence_check_answer"], "YES")
        self.assertEqual(result["followup_answer"], "KUHL")
        self.assertEqual(result["request_bytes"], 250)
        self.assertEqual(result["api_latency_ms"], 300.0)

    def test_skips_ocr_call_when_presence_is_uncertain(self):
        with patch(
            "baseline.ask_deepseek",
            return_value=("UNCERTAIN", 0.1, 100)
        ) as ask_deepseek:
            baseline.run_baseline(
                "What text is on the refrigerator?",
                image_paths=[self.image_path],
                pipeline_results=self.pipeline_results()
            )

        self.assertEqual(ask_deepseek.call_count, 1)
        result = self.load_baseline_results()[0]
        self.assertIn("OCR was skipped", result["answer"])

    def attribute_pipeline_results(self):
        return [{
            "image": "sample.jpg",
            "task_plan": {
                "tasks": [{
                    "task_type": "attribute",
                    "target": "bus",
                    "attribute": "color"
                }]
            },
            "task_results": [{
                "semantic_type": "attribute",
                "target_class": "bus"
            }]
        }]

    def test_skips_attribute_question_when_target_is_absent(self):
        with patch(
            "baseline.ask_deepseek",
            return_value=("NO", 0.1, 100)
        ) as ask_deepseek:
            baseline.run_baseline(
                "What color is the bus?",
                image_paths=[self.image_path],
                pipeline_results=self.attribute_pipeline_results()
            )

        self.assertEqual(ask_deepseek.call_count, 1)
        result = self.load_baseline_results()[0]
        self.assertEqual(result["answer"], "No bus is visible.")
        self.assertEqual(result["targeted_task_kind"], "attribute")
        self.assertIsNone(result["followup_answer"])

    def test_asks_attribute_question_only_after_target_is_confirmed(self):
        with patch(
            "baseline.ask_deepseek",
            side_effect=[
                ("YES", 0.1, 100),
                ("The bus is blue.", 0.2, 150)
            ]
        ) as ask_deepseek:
            baseline.run_baseline(
                "What color is the bus?",
                image_paths=[self.image_path],
                pipeline_results=self.attribute_pipeline_results()
            )

        self.assertEqual(ask_deepseek.call_count, 2)
        second_question = ask_deepseek.call_args_list[1].args[1]
        self.assertIn("What color is the bus?", second_question)
        self.assertIn("target 'bus' only", second_question)
        result = self.load_baseline_results()[0]
        self.assertEqual(result["answer"], "The bus is blue.")
        self.assertEqual(result["targeted_task_kind"], "attribute")
        self.assertEqual(result["followup_answer"], "The bus is blue.")


if __name__ == "__main__":
    unittest.main()
