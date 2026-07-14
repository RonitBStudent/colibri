import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_summary_uses_median_and_rates(self):
        runs = [
            {"valid": True, "grounding": {"grounded": 2, "total": 2},
             "metrics": {"elapsed_seconds": 9, "completion_tokens": 30}},
            {"valid": False, "grounding": {"grounded": 1, "total": 2},
             "metrics": {"elapsed_seconds": 3, "completion_tokens": 20}},
            {"valid": True, "grounding": {"grounded": 2, "total": 2},
             "metrics": {"elapsed_seconds": 6, "completion_tokens": 25}},
        ]
        summary = benchmark.summarize_runs(runs)
        self.assertEqual(summary["median_elapsed_seconds"], 6)
        self.assertEqual(summary["median_completion_tokens"], 25)
        self.assertAlmostEqual(summary["median_end_to_end_completion_tok_s"], 25 / 6)
        self.assertAlmostEqual(summary["valid_json_schema_rate"], 2 / 3)
        self.assertAlmostEqual(summary["grounded_quote_rate"], 5 / 6)

    def test_failed_runs_are_excluded_from_metrics(self):
        summary = benchmark.summarize_runs([{"error": "server unavailable"}])
        self.assertEqual(summary["runs_completed"], 0)
        self.assertIsNone(summary["median_elapsed_seconds"])


if __name__ == "__main__":
    unittest.main()
