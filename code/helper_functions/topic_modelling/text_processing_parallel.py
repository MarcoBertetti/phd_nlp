import contractions
import re
import nltk
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
import gc
import logging
import warnings
import os
import time
import math
# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress NLTK download messages
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Number conversion
# ---------------------------------------------------------------------------
NUMBER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000
}

def parse_written_number(phrase: str):
    tokens = phrase.lower().strip().replace("-", " ").split()
    total = 0
    current = 0

    for tok in tokens:
        if tok in NUMBER_MAP:
            val = NUMBER_MAP[tok]
            if val in {100, 1000, 1_000_000}:
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
        if words[i].lower() in NUMBER_MAP:
            j = i
            phrase_components = []
            while j < len(words) and words[j].lower() in NUMBER_MAP:
                phrase_components.append(words[j].lower())
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

# ---------------------------------------------------------------------------
# Text cleaning and lemmatization
# ---------------------------------------------------------------------------
def clean_and_lemmatize(text: str):
    """Full text preprocessing pipeline: expand contractions, normalize numbers, lemmatize."""
    try:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = contractions.fix(text)
        text = convert_written_numbers_to_digits(text)

        # Remove URLs, IPs, and non-alphanumeric chars
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\b\d+(\.\d+)+\b", "", text)
        text = re.sub(r"[^a-z0-9.\s]", "", text.lower())

        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in STOP_WORDS]

        return " ".join(tokens).strip()

    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return "error"

# ---------------------------------------------------------------------------
# CPU management
# ---------------------------------------------------------------------------
def limit_cpu_usage():
    """Lower process priority to reduce CPU contention."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(19)
    except Exception as e:
        logger.warning(f"Could not adjust CPU priority: {e}")

# ---------------------------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------------------------
def process_chunk(texts):
    """Helper to process a batch of texts in a worker process."""
    limit_cpu_usage()
    results = []
    try:
        for text in texts:
            results.append(clean_and_lemmatize(text))
            time.sleep(0.005)
    except Exception as e:
        logger.error(f"Batch failed: {e}")
    gc.collect()
    return results

# ---------------------------------------------------------------------------
# Parallel preprocessing
# ---------------------------------------------------------------------------
def preprocess_text_parallel(df, text_col="text", num_workers=None, batch_size=10_000):
    """
    Parallelized text preprocessing using multiprocessing.
    Applies normalization and lemmatization to text in chunks.
    """
    limit_cpu_usage()
    total_rows = len(df)
    logger.info(f"Starting parallel text preprocessing on {total_rows} rows")

    if total_rows == 0:
        logger.warning("Empty DataFrame provided — skipping preprocessing.")
        df["clean_text"] = ""
        return df

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    logger.info(f"Using {num_workers} workers for parallel processing")

    df[text_col] = df[text_col].fillna("").astype(str)

    processed_texts = []
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{math.ceil(total_rows/batch_size)}")

        batch_texts = df[text_col].iloc[batch_start:batch_end].tolist()
        chunk_size = max(1, len(batch_texts) // (num_workers * 4))
        chunks = [batch_texts[i:i + chunk_size] for i in range(0, len(batch_texts), chunk_size)]

        try:
            with Pool(num_workers) as pool:
                batch_processed = pool.map(process_chunk, chunks)
                processed_texts.extend([text for sublist in batch_processed for text in sublist])
        except Exception as e:
            logger.error(f"Error in parallel batch: {e}")
            # fallback to sequential
            processed_texts.extend([clean_and_lemmatize(t) for t in batch_texts])

        gc.collect()

    if len(processed_texts) != total_rows:
        logger.warning(f"Processed {len(processed_texts)} rows, expected {total_rows}")

    df["clean_text"] = processed_texts[:total_rows]
    logger.info("Text preprocessing completed successfully")
    gc.collect()
    return df
