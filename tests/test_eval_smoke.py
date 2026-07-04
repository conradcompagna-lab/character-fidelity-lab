import unittest

from character_lab.models import AnswerResult, LoreChunk, RetrievedChunk
from character_lab.scorer import score_case


class EvalSmokeTests(unittest.TestCase):
    def test_answer_and_score(self):
        chunk = LoreChunk(
            id="maelor_bio:001",
            source_id="maelor_bio",
            source_file="maelor.md",
            title="Maelor",
            text="Maelor did not betray the Sunken Temple.",
            start_line=1,
            end_line=1,
        )
        result = AnswerResult(
            question="Did Maelor betray the Sunken Temple?",
            answer="No. Maelor did not betray the Sunken Temple. [maelor_bio]",
            provider="test",
            model="test",
            retrieved=[RetrievedChunk(chunk=chunk, score=1.0)],
        )
        case = {
            "id": "x",
            "prompt": "Did Maelor betray the Sunken Temple?",
            "expected_answer_contains": ["did not betray"],
            "required_source_ids": ["maelor_bio"],
            "should_refuse": False,
        }
        scored = score_case(case, result)
        self.assertGreaterEqual(scored["scores"]["total"], 0.8)


if __name__ == "__main__":
    unittest.main()
