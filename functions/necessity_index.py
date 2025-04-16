import re

# ---------------- MARKERS ----------------

LEXICON = {
    "urgency_markers": [
        "urgent", "need", "shortage", "struggling", "challenge",
        "emergency", "tight", "limited", "difficulties", "crisis",
        "immediate", "critical", "necessary", "essential", "dire"
    ],
    "severity_markers": [
        "trauma","difficult", "violence", "profound", "severe", "extreme",
        "desperate", "worst", "suffering", "devastating", "harsh",
    ],
    "vulnerability_markers": [
        "asd", "send", "disability", "special needs", "diagnosis", "vulnerable",
        "fragile", "risk", "sen", 'adhd','add','dyslexia', 'trans', 'queer',
        'lgbtq', 'refugee', 'refugees', 'autism', 'autisitc', 'neurodivergent',
        'low income', "poverty", "deprived", "poor", "disadvantaged", 'underserved',
        'therapy', 'therapeutic', "aln"
    ],
    "emotional_appeal": [
        "help", "support", "deserve", "hope", "lives", "transform",
        "improve", "amazing", "difference", "dream", "opportunity",
        "empower", "nurture", "change", "impact", "grateful", "please",
        "!", 'passion', 'passionate', 'committeed', 'life-changing'
    ],
    "superlatives": [
        "most", "every", "all", "huge", "massive", "dramatically",
        "significantly", "really", "very", "extremely", "entirely",
        "absolutely", "completely", "totally", "utterly"
    ]
}


# --------------- WEIGHTS --------------

WEIGHTS = {
    'urgency_markers': 2.5,
    'severity_markers': 2.0,
    'vulnerability_markers': 3,
    'emotional_appeal': 1.5,
    'superlatives': 1.0
}


# ------------ FUNCTION ----------------

def compute_necessity(text):

    if not isinstance(text, str):
        return 0.0

    text_lower = text.lower()

    score = 0.0
    for category, keywords in LEXICON.items():
        # For each keyword, count how many times it appears in text
        # (simple usage of re.findall)
        category_count = 0
        for kw in keywords:
            # Escape special characters in kw to ensure correct regex matching
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = re.findall(pattern, text_lower)
            category_count += len(matches)

        # Weight the occurrences by the category weighting
        score += WEIGHTS[category] * category_count

    return score
