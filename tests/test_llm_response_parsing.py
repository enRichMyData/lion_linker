import json
import unittest

from lion_linker.lion_linker import LionLinker


class LLMResponseParsingTests(unittest.TestCase):
    def setUp(self):
        self.lion = LionLinker.__new__(LionLinker)
        self.lion.match_confidence_threshold = 0.5
        self.lion.ranking_size = 5
        self.lion.RANKING_KEY = LionLinker.RANKING_KEY
        self.lion.RANKING_SCORE_PRECISION = LionLinker.RANKING_SCORE_PRECISION

    def test_parse_llm_json_valid_top5(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q42", "confidence_label": "HIGH", "confidence_score": 0.92},
                    {"id": "Q123", "confidence_label": "MEDIUM", "confidence_score": 0.63},
                    {"id": "Q999", "confidence_label": "LOW", "confidence_score": 0.31},
                    {"id": "Q777", "confidence_label": "LOW", "confidence_score": 0.24},
                    {"id": "Q888", "confidence_label": "LOW", "confidence_score": 0.18},
                ],
                "nil_score": 0.12,
            }
        )

        ranking, nil_score = self.lion._parse_llm_json(response)

        self.assertEqual(5, len(ranking))
        self.assertAlmostEqual(0.12, nil_score, places=2)
        self.assertEqual(
            {"id": "Q42", "confidence_label": "HIGH", "confidence_score": 0.92},
            ranking[0],
        )

    def test_parse_llm_json_sorts_by_confidence_score(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q2", "confidence_label": "LOW", "confidence_score": 0.4},
                    {"id": "Q1", "confidence_label": "HIGH", "confidence_score": 0.9},
                ]
            }
        )

        ranking, nil_score = self.lion._parse_llm_json(response)

        self.assertIsNone(nil_score)
        self.assertEqual(["Q1", "Q2"], [entry["id"] for entry in ranking])

    def test_parse_llm_json_allows_nil_entry(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "NIL", "confidence_label": "HIGH", "confidence_score": 0.8}
                ]
            }
        )

        ranking, nil_score = self.lion._parse_llm_json(response)

        self.assertEqual(1, len(ranking))
        self.assertIsNone(nil_score)
        self.assertEqual("NIL", ranking[0]["id"])

    def test_parse_llm_json_requires_confidence_score(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q1", "confidence_label": "HIGH"}
                ]
            }
        )

        with self.assertRaises(ValueError):
            self.lion._parse_llm_json(response)

    def test_parse_llm_json_rejects_unexpected_keys(self):
        response = json.dumps(
            {
                "candidate_ranking": [],
                "answer": "Q42",
            }
        )

        with self.assertRaises(ValueError):
            self.lion._parse_llm_json(response)

    def test_determine_predicted_identifier_requires_high_confidence(self):
        entries = [
            {"id": "Q1", "confidence_label": "HIGH", "confidence_score": 0.7}
        ]
        self.assertEqual("Q1", self.lion._determine_predicted_identifier(entries, None))

        entries_low = [
            {"id": "Q1", "confidence_label": "LOW", "confidence_score": 0.3}
        ]
        self.assertEqual(
            "NIL", self.lion._determine_predicted_identifier(entries_low, None)
        )

        entries_nil = [
            {"id": "NIL", "confidence_label": "HIGH", "confidence_score": 0.9}
        ]
        self.assertEqual(
            "NIL", self.lion._determine_predicted_identifier(entries_nil, None)
        )

    def test_enrich_candidate_ranking_adds_metadata(self):
        entries = [
            {"id": "Q1", "confidence_label": "HIGH", "confidence_score": 0.55},
            {"id": "Q2", "confidence_label": "MEDIUM", "confidence_score": 0.45},
        ]
        candidates = [
            {
                "id": "Q1",
                "name": "Alpha",
                "description": "First",
                "types": [{"name": "Person"}],
            },
            {
                "id": "Q2",
                "name": "Beta",
                "description": "Second",
                "types": [{"name": "Organization"}],
            },
        ]

        predicted = self.lion._determine_predicted_identifier(entries, None)
        enriched = self.lion._enrich_candidate_ranking(entries, candidates, predicted)

        self.assertEqual("Q1", predicted)
        self.assertTrue(enriched[0]["match"])
        self.assertFalse(enriched[1]["match"])
        self.assertEqual([{"id": "", "name": "Person"}], enriched[0]["types"])
        self.assertEqual("Alpha", enriched[0]["name"])

    def test_enrich_candidate_ranking_handles_nil_top(self):
        entries = [
            {"id": "NIL", "confidence_label": "HIGH", "confidence_score": 0.9},
            {"id": "Q1", "confidence_label": "MEDIUM", "confidence_score": 0.6},
        ]

        predicted = self.lion._determine_predicted_identifier(entries, None)
        enriched = self.lion._enrich_candidate_ranking(entries, [], predicted)

        self.assertEqual("NIL", predicted)
        self.assertTrue(enriched[0]["match"])
        self.assertEqual("NIL", enriched[0]["id"])
        self.assertFalse(any(entry["match"] for entry in enriched[1:]))


if __name__ == "__main__":
    unittest.main()
