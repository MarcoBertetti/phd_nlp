"""
geo_extraction.py
-----------------
Lightweight module for extracting geographic entities (locations) from text
using Named Entity Recognition (NER) with spaCy.

Intended for use in both GDELT Events and GKG article pipelines.

Usage:
    from helper_functions.geo_extraction import extract_geo_features

    df = extract_geo_features(df, text_col="clean_text")
"""

import pandas as pd
import spacy

# ---------------------------------------------------------------------
# Load SpaCy model once globally
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "textcat"])
except OSError:
    raise RuntimeError(
        "SpaCy model 'en_core_web_sm' not found. Install with:\n"
        "    python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------
# Location extraction functions
# ---------------------------------------------------------------------
def extract_all_locations(text: str):
    """
    Extract all named locations (GPE or LOC) from text.

    Args:
        text (str): Input text.
    Returns:
        list[str]: List of detected location names.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.label_ in {"GPE", "LOC"}]


def extract_primary_location(text: str):
    """
    Extract the first mentioned location (GPE or LOC) in text.

    Args:
        text (str): Input text.
    Returns:
        str or None: First detected location, or None if not found.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            return ent.text.strip()
    return None


def extract_geo_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add NER-based geolocation columns to a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a text column.
        text_col (str): Name of the column containing text.

    Returns:
        pd.DataFrame: Copy of df with 'all_locations' and 'primary_location' columns.
    """
    df = df.copy()

    print("🔍 Extracting locations using spaCy NER...")
    df["all_locations"] = df[text_col].apply(extract_all_locations)
    df["primary_location"] = df[text_col].apply(extract_primary_location)
    print("✅ Location extraction complete.")
    print("📍 Example extracted locations:")
    print(df["primary_location"].dropna().head(10))

    return df
