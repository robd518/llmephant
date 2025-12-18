import re

def normalize_fact(text: str) -> str:
    """
    Normalize a fact string for deduplication and clarity:
    - trim + collapse whitespace
    - normalize pronouns (my → the user's)
    - strip trailing punctuation
    - fix casing
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    lower = f" {text.lower()} "
    replacements = {
        " my ": " the user's ",
        " i ": " the user ",
        " me ": " the user ",
    }
    for k, v in replacements.items():
        lower = lower.replace(k, v)

    text = lower.strip()
    text = re.sub(r"[.!?]+$", "", text)

    # Capitalize for readability
    if text:
        text = text[0].upper() + text[1:]

    return text