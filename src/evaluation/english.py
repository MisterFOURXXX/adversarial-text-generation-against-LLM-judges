import re

_FALLBACK_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take",
}


def _fallback(text: str) -> float:
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _FALLBACK_WORDS)
    return min(1.0, max(0.0, (hits / len(words)) * 1.5))


def detect_english(text: str) -> float:
    try:
        from lingua import Language, LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.FRENCH, Language.GERMAN,
            Language.SPANISH, Language.DUTCH, Language.ITALIAN,
        ).build()
        if detector.detect_language_of(text) != Language.ENGLISH:
            return 0.0
        for lang, conf in detector.compute_language_confidence_values(text):
            if lang == Language.ENGLISH:
                return max(0.0, min(1.0, conf))
        return 0.0
    except Exception:
        return _fallback(text)