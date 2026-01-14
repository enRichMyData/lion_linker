import unittest

from app.models.jobs import PredictionSummary


class PredictionSummaryTests(unittest.TestCase):
    def test_serializes_structured_answer(self):
        payload = [
            {"id": "Q1", "confidence_label": "HIGH", "confidence_score": 0.9}
        ]
        ps = PredictionSummary(
            column="buyer",
            answer=payload,
            identifier="Q1",
        )
        dumped = ps.model_dump(by_alias=True)
        self.assertIn("answer", dumped)
        self.assertEqual(payload, dumped["answer"])


if __name__ == "__main__":
    unittest.main()
