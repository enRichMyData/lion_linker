import json
import unittest

from app.models.jobs import PredictionSummary


class PredictionSummaryTests(unittest.TestCase):
    def test_serializes_with_parsed_answer(self):
        payload = [
            {"id": "Q1", "confidence_label": "HIGH", "confidence_score": 0.9}
        ]
        ps = PredictionSummary(
            column="buyer",
            answer=json.dumps(payload),
            identifier="Q1",
            parsedAnswer=payload,
        )
        dumped = ps.model_dump(by_alias=True)
        self.assertIn("parsedAnswer", dumped)
        self.assertEqual(payload, dumped["parsedAnswer"])


if __name__ == "__main__":
    unittest.main()
