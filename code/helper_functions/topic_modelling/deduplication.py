import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

def jaccard_similarity(text1, text2):
    """
    Compute Jaccard similarity between two strings.
    """
    set1, set2 = set(text1.split()), set(text2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def deduplicate_by_jaccard(df, text_col='clean_text', threshold=0.9, batch_size=1000):
    """
    Remove near-duplicate texts using Jaccard similarity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'clean_text' column.
    text_col : str
        Column name with preprocessed text.
    threshold : float
        Jaccard similarity threshold (e.g., 0.9 means keep only one among texts ≥ 90% similar).
    batch_size : int
        Compare items in chunks to control memory usage.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    df = df.copy().reset_index(drop=True)
    texts = df[text_col].astype(str).tolist()
    keep_indices = []
    removed = set()

    print(f"Running Jaccard deduplication on {len(texts):,} texts (threshold={threshold})")

    for i in tqdm(range(len(texts)), desc="Deduplicating"):
        if i in removed:
            continue
        keep_indices.append(i)
        base_tokens = set(texts[i].split())
        if not base_tokens:
            continue
        for j in range(i + 1, min(i + batch_size, len(texts))):
            if j in removed:
                continue
            other_tokens = set(texts[j].split())
            if not other_tokens:
                continue
            inter = len(base_tokens & other_tokens)
            union = len(base_tokens | other_tokens)
            if union == 0:
                continue
            if inter / union >= threshold:
                removed.add(j)

    print(f"Removed {len(removed):,} duplicates ({len(keep_indices)} kept).")
    return df.iloc[keep_indices].reset_index(drop=True)

from datasketch import MinHash, MinHashLSH
import re
import pandas as pd
from tqdm import tqdm

def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def deduplicate_minhash(df, text_col='clean_text', threshold=0.9, num_perm=64):
    """Global near-duplicate removal using MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    keep = []
    for idx, text in tqdm(df[text_col].items(), desc="Indexing for dedup"):
        m = MinHash(num_perm=num_perm)
        for token in normalize(text).split():
            m.update(token.encode('utf8'))
        dup = lsh.query(m)
        if not dup:           # new unique text
            lsh.insert(str(idx), m)
            minhashes[idx] = m
            keep.append(idx)
    print(f"Kept {len(keep)} / {len(df)} unique articles "
          f"({len(df)-len(keep)} removed).")
    return df.loc[keep].reset_index(drop=True)

import spacy
from tqdm.notebook import tqdm
import pandas as pd

# Load the lightweight model
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
nlp.max_length = 2_000_000  # avoid length errors for long texts

def extract_admin_from_docs(texts, batch_size=100):
    admin0, admin1, admin2 = [], [], []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Running NER"):
        gpe = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        loc = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
        all_places = gpe + loc
        admin0.append(all_places[0] if len(all_places) > 0 else None)
        admin1.append(all_places[1] if len(all_places) > 1 else None)
        admin2.append(all_places[2] if len(all_places) > 2 else None)
    return admin0, admin1, admin2
    
