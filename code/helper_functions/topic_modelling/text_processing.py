###############################################################################
# 2) Text Preprocessing and Number Context Extraction
###############################################################################
# Basic mapping for single words:
#  - For phrases like "twenty three thousand five hundred," see the parsing function below.

import contractions
import re
import nltk
import pandas as pd

# Download resources if not already available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

NUMBER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1000000
}

def parse_written_number(phrase: str):
    tokens = phrase.lower().strip().replace("-", " ").split()
    total = 0
    current = 0

    for tok in tokens:
        if tok in NUMBER_MAP:
            val = NUMBER_MAP[tok]
            if val in {100, 1000, 1000000}:  
                current = max(1, current) * val
            else:
                current += val
        else:
            return None

    return total + current if total + current > 0 else None

def convert_written_numbers_to_digits(text: str):
    words = text.split()
    converted_words = []
    i = 0

    while i < len(words):
        if words[i] in NUMBER_MAP:
            j = i
            phrase_components = []
            while j < len(words) and words[j] in NUMBER_MAP:
                phrase_components.append(words[j])
                j += 1

            phrase_str = " ".join(phrase_components)
            value = parse_written_number(phrase_str)

            if value is not None:
                converted_words.append(str(value))
                i = j
                continue

        converted_words.append(words[i])
        i += 1

    return " ".join(converted_words)

def clean_and_lemmatize(text):
    try:
        text = contractions.fix(text)
        text = convert_written_numbers_to_digits(text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove IP addresses and numeric-dot patterns
        text = re.sub(r'\b\d+(\.\d+)+\b', '', text)

        # Remove non-alphanumeric characters except dots and spaces
        text = re.sub(r'[^a-z0-9.\s]', '', text.lower())

        # Tokenization & Lemmatization (NLTK - lightweight)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in STOP_WORDS]

        return " ".join(tokens)
    
    except Exception as e:
        print(f"🚨 Error processing text: {repr(text[:200])}... (truncated)")
        print(f"Error message: {e}")
        return "Error Error"

def preprocess_text(df, text_col='text'):
    df[text_col] = df[text_col].fillna("").astype(str)
    df['clean_text'] = df[text_col].apply(clean_and_lemmatize)
    return df
