import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import extract


SCHEMA = json.loads((ROOT / "paper.schema.json").read_text())
VALID = {
    "title": "Study",
    "research_question": "Do breaks improve recall?",
    "methods": [{"name": "Randomized study", "description": "Two schedules"}],
    "datasets": [{"name": "Student sample", "size": "60", "population": "Students"}],
    "claims": [{
        "claim": "Breaks improved delayed recall.",
        "evidence_quote": "the structured-break group averaged 13.6",
        "interpretation": "The delayed score was higher.",
        "strength": "strong",
    }],
    "limitations": ["One school"],
}


class FakeResponse:
    def __init__(self, body, headers=None):
        self.body = json.dumps(body).encode()
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


class ExtractTests(unittest.TestCase):
    def test_endpoint_normalization(self):
        expected = "http://localhost:8000/v1/chat/completions"
        self.assertEqual(extract.chat_completions_url("http://localhost:8000"), expected)
        self.assertEqual(extract.chat_completions_url("http://localhost:8000/v1"), expected)
        self.assertEqual(extract.chat_completions_url(expected), expected)

    def test_schema_accepts_expected_shape(self):
        self.assertEqual(extract.validate_schema(VALID, SCHEMA), [])

    def test_schema_reports_missing_and_extra_properties(self):
        value = dict(VALID)
        value.pop("limitations")
        value["invented"] = True
        errors = extract.validate_schema(value, SCHEMA)
        self.assertTrue(any("missing required property 'limitations'" in error for error in errors))
        self.assertTrue(any("unexpected property 'invented'" in error for error in errors))

    def test_grounding_requires_quote_in_source(self):
        report = extract.grounding_report(
            VALID, "Results: the structured-break group averaged 13.6 after seven days."
        )
        self.assertEqual(report["grounded"], 1)
        self.assertEqual(report["total"], 1)

    def test_analyze_paper_parses_openai_response(self):
        response = {
            "choices": [{"message": {"content": json.dumps(VALID)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        paper = "Results: the structured-break group averaged 13.6 after seven days."
        with patch("extract.urlopen", return_value=FakeResponse(
            response, {"x-colibri-queue-wait-ms": "7"}
        )):
            result = extract.analyze_paper(paper, SCHEMA, "http://localhost:8000/v1", "test")
        self.assertTrue(result["valid"])
        self.assertEqual(result["grounding"]["grounded"], 1)
        self.assertEqual(result["metrics"]["completion_tokens"], 50)
        self.assertEqual(result["metrics"]["queue_wait_ms"], "7")

    def test_markdown_fence_is_tolerated_but_not_required(self):
        parsed = extract.parse_json_content("```json\n{\"ok\": true}\n```")
        self.assertEqual(parsed, {"ok": True})


if __name__ == "__main__":
    unittest.main()
