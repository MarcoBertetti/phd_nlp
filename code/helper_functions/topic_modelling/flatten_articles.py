###############################################################################
# 1) Flatten articles: one row per (header, body) pair
###############################################################################
import numpy as np
import pandas as pd
import re

def flatten_articles(df, articles_col="articles"):
    """
    Explode ONLY 'articles' column.
    For ALL other columns:
      - If value is a list with same length as 'articles', pick corresponding element.
      - If value is a single-element list, flatten it.
      - Otherwise leave as-is.
    """
    df = df.reset_index(drop=True).reset_index(names="original_idx")

    # keep rows that actually have article lists
    df = df[df[articles_col].apply(lambda x: isinstance(x, list))]

    # record name of all other columns
    other_cols = [c for c in df.columns if c != articles_col]

    # length of each article list
    lengths = df[articles_col].apply(len)

    # Construct per-article rows
    rows = []
    for idx, row in df.iterrows():
        art_list = row[articles_col]
        n = len(art_list)

        for i in range(n):
            new_row = {"original_idx": row["original_idx"], articles_col: str(art_list[i])}

            for col in other_cols:
                val = row[col]

                if isinstance(val, list):
                    if len(val) == n:          # match by index
                        new_row[col] = val[i]
                    elif len(val) == 1:        # flatten 1-element lists
                        new_row[col] = val[0]
                    else:                      # leave as-is
                        new_row[col] = val
                else:
                    new_row[col] = val

            rows.append(new_row)

    return pd.DataFrame(rows)

###############################################################################
# 2) Filter articles by LEAP4FNSSA lexicon
###############################################################################
def filter_articles_by_lexicon(
    df,
    clean_text_col="clean_text",
    lexicon_path="LEAP4FNSSA_LEXICON_long.csv",
    include_concepts=True
):
    """
    Filters articles whose `clean_text` contains at least one keyword or concept
    term from the given lexicon CSV.
    - include_concepts: if True, also match 'Concept' column values
    """

    # Robust CSV loading
    lexicon = pd.read_csv(lexicon_path, quotechar='"', encoding="utf-8", on_bad_lines="skip")

    # Detect columns
    if "Keyword" not in lexicon.columns:
        raise ValueError(f"Lexicon must have a 'Keyword' column. Found: {lexicon.columns.tolist()}")
    if include_concepts and "Concept" not in lexicon.columns:
        print("⚠️ 'Concept' column not found — only using 'Keyword'.")

    # Collect keyword list
    keywords = lexicon["Keyword"].astype(str).str.strip().str.lower().unique().tolist()

    # Optionally also include concept labels (e.g. 'humanitarian response')
    if include_concepts and "Concept" in lexicon.columns:
        keywords += (
            lexicon["Concept"].astype(str).str.strip().str.lower().unique().tolist()
        )

    # Clean and deduplicate
    keywords = [k for k in keywords if len(k) > 1 and k not in ("nan", "none", "")]
    keywords = sorted(set(keywords))

    print(f"🔍 Filtering {len(df):,} articles using {len(keywords):,} lexicon keywords...")

    # Compile regex for all keywords (case-insensitive, word boundaries)
    pattern = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, keywords)), flags=re.IGNORECASE)

    # Apply vectorized search
    mask = df[clean_text_col].astype(str).str.contains(pattern, na=False, regex=True)
    filtered_df = df[mask].reset_index(drop=True)

    print(f"✅ Retained {len(filtered_df):,} / {len(df):,} articles "
          f"({len(filtered_df)/len(df)*100:.2f}%) after lexicon filtering.")
    return filtered_df
