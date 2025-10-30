import json
import unittest

from lion_linker.lion_linker import LionLinker


class LLMResponseParsingTests(unittest.TestCase):
    def test_parse_llm_json_valid_top5(self):
        response = json.dumps(
            {
                "answer": "Q42",
                "candidate_ranking": [
                    {"rank": 1, "id": "Q42", "score": 0.95001, "label": "Douglas Adams"},
                    {"rank": 2, "id": "Q123", "score": 0.74, "label": "Another Candidate"},
                    {"rank": 3, "id": "Q999", "score": 0.51, "label": "Third"},
                ],
            }
        )

        payload_str, answer, ranking_str, normalized_entries = LionLinker._parse_llm_json(
            response, 5
        )

        self.assertEqual('{"answer":"Q42","candidate_ranking":[{"rank":1,"id":"Q42","score":0.95,"label":"Douglas Adams"},{"rank":2,"id":"Q123","score":0.74,"label":"Another Candidate"},{"rank":3,"id":"Q999","score":0.51,"label":"Third"}]}', payload_str)
        self.assertEqual("Q42", answer)
        self.assertEqual(
            '{"candidate_ranking":[{"rank":1,"id":"Q42","score":0.95,"label":"Douglas Adams"},'
            '{"rank":2,"id":"Q123","score":0.74,"label":"Another Candidate"},'
            '{"rank":3,"id":"Q999","score":0.51,"label":"Third"}]}',
            ranking_str,
        )
        self.assertEqual(3, len(normalized_entries))

    def test_parse_llm_json_enforces_score_bounds(self):
        response = json.dumps(
            {
                "answer": "Q42",
                "candidate_ranking": [
                    {"rank": 1, "id": "Q42", "score": 1.0, "label": "Douglas Adams"},
                ],
            }
        )

        with self.assertRaises(ValueError):
            LionLinker._parse_llm_json(response, 5)

    def test_parse_llm_json_enforces_strict_rank_sequence(self):
        response = json.dumps(
            {
                "answer": "Q42",
                "candidate_ranking": [
                    {"rank": 1, "id": "Q42", "score": 0.9, "label": "Douglas Adams"},
                    {"rank": 3, "id": "Q123", "score": 0.7, "label": "Second"},
                ],
            }
        )

        with self.assertRaises(ValueError):
            LionLinker._parse_llm_json(response, 5)

    def test_parse_llm_json_missing_label_rejected(self):
        response = json.dumps(
            {
                "answer": "Q42",
                "candidate_ranking": [
                    {"rank": 1, "id": "Q42", "score": 0.9},
                ],
            }
        )

        with self.assertRaises(ValueError):
            LionLinker._parse_llm_json(response, 5)

    def test_parse_llm_json_rejects_unsupported_top_k(self):
        response = json.dumps({"answer": "Q42", "candidate_ranking": []})
        with self.assertRaises(ValueError):
            LionLinker._parse_llm_json(response, 4)


if __name__ == "__main__":
    unittest.main()
