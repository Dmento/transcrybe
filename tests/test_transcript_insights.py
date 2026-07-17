import unittest

from transcript_utils import build_transcript_insights


class TranscriptInsightsTests(unittest.TestCase):
    def test_build_transcript_insights_counts_words_and_keywords(self):
        transcript = (
            "Machine learning helps teams analyze data quickly. "
            "Teams use machine learning to automate routine tasks."
        )

        insights = build_transcript_insights(transcript)

        self.assertEqual(insights["word_count"], 15)
        self.assertEqual(insights["estimated_duration_minutes"], 1)
        self.assertIn("machine", insights["keywords"])
        self.assertIn("learning", insights["keywords"])
        self.assertIn("teams", insights["keywords"])

    def test_build_transcript_insights_handles_empty_text(self):
        insights = build_transcript_insights("")

        self.assertEqual(insights["word_count"], 0)
        self.assertEqual(insights["estimated_duration_minutes"], 0)
        self.assertEqual(insights["keywords"], [])
        self.assertEqual(insights["preview"], "")


if __name__ == "__main__":
    unittest.main()
