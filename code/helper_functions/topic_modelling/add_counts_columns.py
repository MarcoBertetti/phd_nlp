import re
import pandas as pd
import numpy as np
import spacy
import swifter
nlp = spacy.load("en_core_web_sm")

# Predefine FS_CATEGORIES
FS_CATEGORIES = {
    "fatalities": [
        "killed", "dead", "death", "deaths", "fatalities", "casualties", "life", "lost", "loss",
        "massacred", "loss of life", "lost their lives", "body discover", "tragically"
    ],
    "displaced": [
        "displaced", "displacement", "fled", "evacuated", "refugees", "flee", "uprooted", "seek refuge",
        "leave", "relocation", "relocated", "relocate"
    ],
    "detained": [
        "detained", "arrested", "imprisoned", "incarcerated", "held in custody", "held"
    ],
    "injured": [
        "injured", "wounded", "hurt"
    ],
    "sexual_violence": [
        "rape", "raped", "sexual violence", "sexual assault", "sexually assaulted", "sexual harassment"
    ],
    "torture": [
        "torture", "tortured"
    ],
    "economic_shocks": [
        "economic shock", "economic shocks", "financial crisis", "market collapse", "recession", "depression"
    ],
    "agriculture": [
        "agriculture", "agricultural", "farming", "crop", "harvest", "livestock", "agrarian"
    ],
    "weather": [
        "weather", "drought", "flood", "flooding", "storm", "rainfall", "extreme weather", "heatwave",
        "cold snap", "hurricane", "cyclone", "typhoon", "wildfire"
    ],
    "food_insecurity": [
        "food insecurity", "food shortage", "malnutrition", "starvation", "undernourished", "lack of food", "food crisis"
    ]
}

# **Precompile regex patterns**
FS_PATTERNS = {cat: re.compile(r'\b(?:' + '|'.join(map(re.escape, synonyms)) + r')\b', re.IGNORECASE)
               for cat, synonyms in FS_CATEGORIES.items()}

def count_synonym_frequencies(text: str) -> dict:
    """
    Given a text string, return a dictionary:
      { "<category>_freq": <count of matches>, ... }
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {f"{cat}_freq": 0 for cat in FS_CATEGORIES}

    text = re.sub(r'\.(?!\d)', '', text.lower().strip())  # Keep decimal points (e.g., 3.14)

    return {f"{cat}_freq": len(pattern.findall(text)) for cat, pattern in FS_PATTERNS.items()}

def add_synonym_frequency_columns(df: pd.DataFrame, text_col: str = 'clean_text') -> pd.DataFrame:
    """
    For each row in df, count how many times each category appears.
    Adds new columns (e.g., "fatalities_freq", "weather_freq", etc.) to df.
    """
    freq_df = df[text_col].swifter.apply(lambda txt: pd.Series(count_synonym_frequencies(txt)))
    return pd.concat([df, freq_df], axis=1)

###############################################################################
# 5) Extract numbers and associate them with the closest FS category
###############################################################################
# List of categories to include
INCLUDED_CATEGORIES = [
    "fatalities", "displaced", "detained", "injured", "sexual_violence", "torture"
]

# Define singular nouns for inference
SINGULAR_NOUNS = {
    "fatalities": {"child", "girl", "boy", "person", "man", "woman", "baby", "passenger"},
    "displaced": {"refugee", "evacuee", "migrant"},
    "detained": {"prisoner", "detainee"},
    "injured": {"victim", "patient"},
    "sexual_violence": {"victim", "survivor"},
    "torture": {"prisoner", "detainee"}
}

# Define time-related units
TIME_UNITS = {"hour", "hours", "minute", "minutes", "second", "seconds",
              "day", "days", "week", "weeks", "month", "months", "year", "years"}

def infer_category_count(subtree, category):
    """Infers if an event has at least 1 affected individual for a given category."""
    if category not in FS_CATEGORIES or category not in SINGULAR_NOUNS:
        return None  # Skip if category is not explicitly mapped

    subtree_texts = {token.text.lower() for token in subtree}
    return 1 if (subtree_texts & FS_CATEGORIES[category]) and (subtree_texts & SINGULAR_NOUNS[category]) else None

def extract_valid_numbers(subtree, category=None):
    """Extracts valid numbers from a subtree, allowing large numbers only for displacement."""
    numbers = []
    tokens = list(subtree)

    for i, token in enumerate(tokens):
        if token.like_num and token.text.replace(",", "").replace(".", "").isdigit():
            try:
                num = float(token.text.replace(",", ""))
                if i + 1 < len(tokens) and tokens[i + 1].text.replace(",", "").isdigit():
                    next_num = float(tokens[i + 1].text.replace(",", ""))
                    if next_num == 1_000_000:
                        num *= 1_000_000  # Convert to millions
                        continue  # Skip next token

                if num >= 1_000_000 and category != "displaced":
                    continue  # Skip large numbers for non-displacement categories

                numbers.append((num, token.i))

            except ValueError:
                pass  # Skip non-numeric values

    return numbers

def find_closest_number(token_idx, numbers):
    """Finds the closest number to a given token index."""
    if not numbers:
        return None
    indices = np.array([idx for _, idx in numbers])
    return numbers[np.argmin(np.abs(indices - token_idx))][0]

def count_category_mentions(text):
    """Processes a text string to map INCLUDED_CATEGORIES mentions to their closest numerical value."""
    if pd.isna(text) or not isinstance(text, str):
        return {f"{cat}_count": 0 for cat in INCLUDED_CATEGORIES}

    # Truncate text to avoid SpaCy max_length issues
    text = text[:100_000]  

    doc = nlp(text)
    category_counts = {f"{cat}_count": 0 for cat in INCLUDED_CATEGORIES}
    max_displaced = 0

    for token in doc:
        if token.head == token:  # Identify the root of a dependency tree
            subtree = list(token.subtree)

            # Step 1: Apply all filtering rules in one pass
            filtered_subtree = [
                t for t in subtree
                if t.ent_type_ not in {"DATE", "TIME"} and not (t.like_num and 
                (t.i + 1 < len(doc) and doc[t.i + 1].text.lower() in TIME_UNITS))
            ]

            # Step 2: Extract valid numbers
            valid_numbers = extract_valid_numbers(filtered_subtree, category="displaced")

            if not filtered_subtree:
                continue

            # Step 3: Match categories and assign numbers
            matched_categories = set()
            for t in filtered_subtree:
                for cat in INCLUDED_CATEGORIES:
                    if cat not in matched_categories and any(
                        s in t.text.lower() for s in FS_CATEGORIES[cat]
                    ):
                        matched_categories.add(cat)
                        closest_num = find_closest_number(t.i, valid_numbers)
                        
                        if cat == "displaced":
                            max_displaced = max(max_displaced, closest_num or 0)
                        else:
                            category_counts[f"{cat}_count"] += closest_num or 1

    category_counts["displaced_count"] = max_displaced
    return category_counts

def add_category_count_columns(df, text_col="clean_text"):
    """Adds category count columns to the DataFrame."""
    count_df = df[text_col].swifter.apply(lambda txt: pd.Series(count_category_mentions(txt)))
    return pd.concat([df, count_df.fillna(0).astype(int)], axis=1)
