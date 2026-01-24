import re
import pandas as pd
import numpy as np
import spacy
import swifter
import os
from joblib import Parallel, delayed
import math
from tqdm.notebook import tqdm

# ---------------------------------------------------------------------------
# Food Security Category Definitions
# ---------------------------------------------------------------------------
FS_CATEGORIES = {
    "fatalities": [
        "killed", "dead", "death", "deaths", "fatalities", "casualties", "life", "lost", "loss",
        "massacred", "loss of life", "lost their lives", "body discover", "tragically"
    ],
    "displaced": [
        "displaced", "displacement", "fled", "evacuated", "refugees", "flee", "uprooted",
        "seek refuge", "leave", "relocation", "relocated", "relocate"
    ],
    "detained": [
        "detained", "arrested", "imprisoned", "incarcerated", "held in custody", "held"
    ],
    "injured": [
        "injured", "wounded", "hurt"
    ],
    "sexual_violence": [
        "rape", "raped", "sexual violence", "sexual assault",
        "sexually assaulted", "sexual harassment"
    ],
    "torture": [
        "torture", "tortured"
    ],
    "economic_shocks": [
        "economic shock", "economic shocks", "financial crisis",
        "market collapse", "recession", "depression"
    ],
    "agriculture": [
        "agriculture", "agricultural", "farming", "crop", "harvest",
        "livestock", "agrarian"
    ],
    "weather": [
        "weather", "drought", "flood", "flooding", "storm", "rainfall",
        "extreme weather", "heatwave", "cold snap", "hurricane", "cyclone",
        "typhoon", "wildfire"
    ],
    "food_insecurity": [
        "food insecurity", "food shortage", "malnutrition", "starvation",
        "undernourished", "lack of food", "food crisis"
    ]
}

# Precompile regex patterns
FS_PATTERNS = {
    cat: re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, synonyms)), re.IGNORECASE)
    for cat, synonyms in FS_CATEGORIES.items()
}

# ---------------------------------------------------------------------------
# Frequency counting
# ---------------------------------------------------------------------------
def count_synonym_frequencies(text: str) -> dict:
    """Count keyword occurrences for each category in a text."""
    if not isinstance(text, str) or not text.strip():
        return {f"{cat}_freq": 0 for cat in FS_CATEGORIES}

    text = re.sub(r"\.(?!\d)", "", text.lower().strip())  # keep decimals
    return {f"{cat}_freq": len(pattern.findall(text)) for cat, pattern in FS_PATTERNS.items()}


def add_synonym_frequency_columns(df: pd.DataFrame, text_col="clean_text") -> pd.DataFrame:
    """Add frequency columns for each FS category."""
    freq_df = df[text_col].swifter.apply(lambda txt: pd.Series(count_synonym_frequencies(txt)))
    return pd.concat([df, freq_df], axis=1)

# ---------------------------------------------------------------------------
# Category–number association logic
# ---------------------------------------------------------------------------
INCLUDED_CATEGORIES = ["fatalities", "displaced", "detained", "injured", "sexual_violence", "torture"]

SINGULAR_NOUNS = {
    "fatalities": {"child", "girl", "boy", "person", "man", "woman", "baby", "passenger"},
    "displaced": {"refugee", "evacuee", "migrant"},
    "detained": {"prisoner", "detainee"},
    "injured": {"victim", "patient"},
    "sexual_violence": {"victim", "survivor"},
    "torture": {"prisoner", "detainee"},
}

TIME_UNITS = {
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "day", "days", "week", "weeks", "month", "months", "year", "years"
}

def extract_valid_numbers(subtree, category=None):
    """Extract numeric values while ignoring temporal quantities."""
    numbers = []
    for i, token in enumerate(subtree):
        if token.like_num:
            try:
                val = float(token.text.replace(",", ""))
                # handle "million" etc.
                if i + 1 < len(subtree) and subtree[i + 1].text == "million":
                    val *= 1_000_000
                if val >= 1_000_000 and category != "displaced":
                    continue
                numbers.append((val, token.i))
            except ValueError:
                continue
    return numbers


def find_closest_number(token_idx, numbers):
    """Return numeric value closest in position to token index."""
    if not numbers:
        return None
    arr = np.array([i for _, i in numbers])
    return numbers[np.argmin(np.abs(arr - token_idx))][0]

def count_category_mentions(doc):
    """Assign nearest numeric magnitude to each category mention using parsed doc."""
    counts = {f"{cat}_count": 0 for cat in INCLUDED_CATEGORIES}
    max_displaced = 0

    for sent in doc.sents:
        tokens = [
            t for t in sent if not (
                (t.like_num and t.i + 1 < len(doc) and doc[t.i + 1].text.lower() in TIME_UNITS)
                or t.ent_type_ in {"DATE", "TIME"}
            )
        ]

        numbers = extract_valid_numbers(tokens, category="displaced")
        sent_text = sent.text.lower()

        for cat in INCLUDED_CATEGORIES:
            if any(kw in sent_text for kw in FS_CATEGORIES[cat]):
                num = find_closest_number(sent.start, numbers)

                if cat == "displaced":
                    max_displaced = max(max_displaced, num or 0)
                else:
                    counts[f"{cat}_count"] += num or 1

    counts["displaced_count"] = max_displaced
    return counts

def add_category_count_columns(df, text_col="clean_text"):
    texts = df[text_col].tolist()

    # spaCy pipe with parallelization and progress bar
    docs = nlp.pipe(
        texts,
        batch_size=200,
        n_process=os.cpu_count(),
        as_tuples=False
    )

    # Wrap docs with tqdm for progress display
    results = [count_category_mentions(doc) for doc in tqdm(docs, total=len(texts))]

    count_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), count_df.fillna(0).astype(int)], axis=1)

# ---------------------------------------------------------------------------
# Optimized spaCy model
# ---------------------------------------------------------------------------
def create_optimized_nlp():
    return spacy.load(
        "en_core_web_sm",
        disable=["ner", "textcat", "entity_linker", "entity_ruler", "lemmatizer"]
    )

nlp = create_optimized_nlp()
