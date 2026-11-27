import tempfile
import unittest
from pathlib import Path

from toolbar.payload_totals import write_evaluation_report


class UnifiedAgreementReportTests(unittest.TestCase):
    def test_exact_ocr_similarity_counts_as_a_match_in_report(self):
        pipeline_results = [{
            "image": "sample.jpg",
            "question": "What text is on the refrigerator?",
            "task_type": "TEXT",
            "route": "CROP",
            "status": "SUCCESS",
            "answer": "KUHL",
            "payload_metrics": {"total_payload_bytes": 100}
        }]
        baseline_results = [{
            "image": "sample.jpg",
            "question": "What text is on the refrigerator?",
            "status": "SUCCESS",
            "answer": "KUHL",
            "payload_metrics": {"total_payload_bytes": 200}
        }]
        comparisons = [{
            "image": "sample.jpg",
            "question": "What text is on the refrigerator?",
            "agreement": 1,
            "accuracy": None,
            "similarity": 1.0,
            "similarity_metric": "normalized_edit"
        }]
        summary = {
            "total_paired_images": 1,
            "evaluated_paired_images": 1,
            "average_payload_reduction_percent": 50.0,
            "pipeline": {
                "average_latency_ms": 10,
                "average_total_payload_bytes": 100
            },
            "baseline": {
                "average_latency_ms": 20,
                "average_total_payload_bytes": 200
            },
            "paired_accuracy": {
                "agreement_accuracy_percent": 100.0,
                "evaluated_images": 1,
                "matched_images": 1,
                "mismatched_images": 0
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "evaluation.txt"
            write_evaluation_report(
                report_path,
                pipeline_results,
                baseline_results,
                comparisons,
                summary
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn(
            "Exact agreement: 100.0% (1 evaluated; 1 MATCH, 0 MISMATCH)",
            report
        )
        self.assertIn(
            "Exact agreement: MATCH (OCR normalized edit similarity: 100.00%)",
            report
        )
        self.assertNotIn("Average OCR normalized edit similarity:", report)


if __name__ == "__main__":
    unittest.main()
