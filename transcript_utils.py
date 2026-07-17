import re
from collections import Counter


def build_transcript_insights(transcript: str) -> dict:
    """Return lightweight insights for a transcript.

    The result includes a word count, a rough duration estimate, a short preview,
    and the most common content words (excluding stop words).
    """
    text = (transcript or "").strip()
    if not text:
        return {
            "word_count": 0,
            "estimated_duration_minutes": 0,
            "keywords": [],
            "preview": "",
        }

    words = re.findall(r"[A-Za-z']+", text.lower())
    word_count = len(words)
    estimated_duration_minutes = max(1, round(word_count / 140))

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "had", "has", "have", "he", "her", "here", "his", "in", "into", "is",
        "it", "its", "of", "on", "or", "our", "that", "the", "their", "there",
        "they", "this", "to", "was", "were", "what", "when", "where", "which",
        "who", "will", "with", "you", "your"
    }

    counts = Counter(word for word in words if word not in stop_words and len(word) > 2)
    keywords = [word for word, _ in counts.most_common(5)]

    preview = " ".join(text.split())
    if len(preview) > 220:
        preview = preview[:217].rstrip() + "..."

    return {
        "word_count": word_count,
        "estimated_duration_minutes": estimated_duration_minutes,
        "keywords": keywords,
        "preview": preview,
    }
