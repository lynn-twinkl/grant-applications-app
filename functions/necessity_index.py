import pandas as pd
import re

# ---------------- MARKERS ----------------

LEXICON = {
    "urgency_markers": [
        "urgent", "need", "shortage", "emergency", "limited", 
        "difficulties", "crisis", "immediate", "critical", "necessary",
        "essential", "dire", "catastrophe"
    ],
    "severity_markers": [
        "trauma","difficult", "profound", "severe", "extreme", "struggling",
        "desperate", "suffering", "devastating", "harsh", "violent", "challenge",
        "danger"
    ],

    "vulnerability_markers": [
        "asd", "send", "disability", "disabilities", "special needs", "diagnosis", "vulnerable",
        "fragile", "risk", "sen", 'adhd','add','dyslexia', 'trans', 'queer',
        'lgbtq', 'refugee', 'refugees', 'autism', 'autisitc', 'neurodivergent',
        'low income', "poverty", "deprived", "poor", "disadvantaged", 'underserved',
        'therapy', 'therapeutic', "aln", "semh", 'violence', 'mental health', 'depressed',
        'anxious', 'anxiety', 'ill', 'sick','down syndrome', 'epilepsy',
    ],
    "emotional_appeal": [
        "help", "support", "deserve", "hope", "lives", "transform",
        "improve", "amazing", "difference", "dream", "opportunity",
        "empower", "nurture", "change", "impact", "grateful", "please",
        "!", 'passion', 'passionate', 'committed', 'life-changing',
        'thank you', 'thankful', 'love'
    ],
    "superlatives": [
        "most", "every", "all", "huge", "massive", "dramatically",
        "significantly", "really", "very", "extremely", "entirely",
        "absolutely", "completely", "totally", "utterly"
    ]
}


# --------------- WEIGHTS --------------

WEIGHTS = {
    'urgency_markers': 3,
    'severity_markers': 2.5,
    'vulnerability_markers': 3,
    'emotional_appeal': 1.5,
    'superlatives': 1.0
}


# ------------ FUNCTION ----------------

def compute_necessity(text):

    if not isinstance(text, str):
        return pd.Series({
            "necessity_index": 0.0,
            "urgency_score": 0.0,
            "severity_score": 0.0,
            "vulnerability_score": 0.0,
        })

    text_lower = text.lower()

    totals = {
            "necessity_index" : 0.0,
            "urgency_score" : 0.0,
            "severity_score" : 0.0,
            "vulnerability_score" : 0.0,
            }


    for category, keywords in LEXICON.items():
        # For each keyword, count how many times it appears in text
        # (simple usage of re.findall)
        category_count = 0
        for kw in keywords:
            # Escape special characters in kw to ensure correct regex matching
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = re.findall(pattern, text_lower)
            category_count += len(matches)
    
        totals['necessity_index'] += WEIGHTS[category] * category_count

        if category == "urgency_markers":
            totals['urgency_score'] += category_count
        elif category == "severity_markers":
            totals['severity_score'] += category_count
        elif category == "vulnerability_markers":
            totals['vulnerability_score'] += category_count

    return pd.Series(totals)
