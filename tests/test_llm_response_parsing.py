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
                    {"id": "Q42", "score": 0.92},
                    {"id": "Q123", "score": 0.63},
                    {"id": "Q999", "score": 0.31},
                    {"id": "Q777", "score": 0.24},
                    {"id": "Q888", "score": 0.18},
                ],
                "nil_score": 0.12,
                "explanation": "Top candidates closely match the mention context.",
            }
        )

        ranking, nil_score, explanation = self.lion._parse_llm_json(response)

        self.assertEqual(5, len(ranking))
        self.assertAlmostEqual(0.12, nil_score, places=2)
        self.assertTrue(explanation)
        self.assertEqual(
            {"id": "Q42", "score": 0.92},
            ranking[0],
        )

    def test_parse_llm_json_sorts_by_score(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q2", "score": 0.4},
                    {"id": "Q1", "score": 0.9},
                ],
                "explanation": "Q1 scored higher confidence than Q2.",
            }
        )

        ranking, nil_score, explanation = self.lion._parse_llm_json(response)

        self.assertIsNone(nil_score)
        self.assertTrue(explanation)
        self.assertEqual(["Q1", "Q2"], [entry["id"] for entry in ranking])

    def test_parse_llm_json_allows_nil_entry(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "NIL", "score": 0.8}
                ],
                "explanation": "No provided candidate matched the context.",
            }
        )

        ranking, nil_score, explanation = self.lion._parse_llm_json(response)

        self.assertEqual(0, len(ranking))
        self.assertIsNone(nil_score)
        self.assertTrue(explanation)
        self.assertEqual("No provided candidate matched the context.", explanation)

    def test_parse_llm_json_requires_score(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q1"}
                ],
                "explanation": "Missing confidence score triggers a failure.",
            }
        )

        with self.assertRaises(ValueError):
            self.lion._parse_llm_json(response)

    def test_parse_llm_json_rejects_unexpected_keys(self):
        response = json.dumps(
            {
                "candidate_ranking": [],
                "answer": "Q42",
                "explanation": "Contains an unexpected key.",
            }
        )

        with self.assertRaises(ValueError):
            self.lion._parse_llm_json(response)

    def test_parse_llm_json_requires_explanation(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "Q1", "score": 0.8}
                ]
            }
        )

        with self.assertRaises(ValueError):
            self.lion._parse_llm_json(response)

    def test_parse_llm_json_handles_nil_score_and_explanation(self):
        response = json.dumps(
            {
                "candidate_ranking": [
                    {"id": "NIL", "score": 0.9},
                    {"id": "Q1", "score": None},
                ],
                "nil_score": 0.88,
                "explanation": "Mention clearly references no known entity.",
            }
        )

        ranking, nil_score, explanation = self.lion._parse_llm_json(response)

        self.assertEqual("Q1", ranking[0]["id"])
        self.assertIsNone(ranking[0]["score"])
        self.assertAlmostEqual(0.88, nil_score, places=2)
        self.assertEqual("Mention clearly references no known entity.", explanation)

    def test_determine_predicted_identifier_requires_high_confidence(self):
        entries = [
            {"id": "Q1", "score": 0.7}
        ]
        self.assertEqual("Q1", self.lion._determine_predicted_identifier(entries, None))

        entries_low = [
            {"id": "Q1", "score": 0.3}
        ]
        self.assertEqual(
            "NIL", self.lion._determine_predicted_identifier(entries_low, None)
        )

        entries_nil = [
            {"id": "NIL", "score": 0.9}
        ]
        self.assertEqual(
            "NIL", self.lion._determine_predicted_identifier(entries_nil, None)
        )

    def test_enrich_candidate_ranking_adds_metadata(self):
        entries = [
            {"id": "Q1", "score": 0.55},
            {"id": "Q2", "score": 0.45},
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
            {"id": "Q1", "score": None},
        ]

        predicted = self.lion._determine_predicted_identifier(entries, None)
        enriched = self.lion._enrich_candidate_ranking(entries, [], predicted)

        self.assertEqual("NIL", predicted)
        self.assertFalse(any(entry["match"] for entry in enriched))


if __name__ == "__main__":
    unittest.main()
