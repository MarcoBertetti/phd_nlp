from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

# Initialize the VADER Sentiment Analyzer once (global)
sid = SentimentIntensityAnalyzer()

def compute_sentiment_scores_in_partition(args):
    """Compute full VADER sentiment scores for a DataFrame partition."""
    partition_df, text_col, show_progress = args
    partition_df = partition_df.copy()

    # Ensure text column exists and is string
    if text_col not in partition_df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")
    partition_df[text_col] = partition_df[text_col].fillna("").astype(str)

    # Apply sentiment analysis
    if show_progress:
        sentiment_scores = [
            sid.polarity_scores(text) if text.strip() else {"compound": 0, "neg": 0, "neu": 0, "pos": 0}
            for text in tqdm(partition_df[text_col], desc="Processing texts", leave=False)
        ]
    else:
        sentiment_scores = [
            sid.polarity_scores(text) if text.strip() else {"compound": 0, "neg": 0, "neu": 0, "pos": 0}
            for text in partition_df[text_col]
        ]

    partition_df["sentiment"] = sentiment_scores

    # Extract individual sentiment scores
    partition_df["compound_score"] = partition_df["sentiment"].apply(lambda x: x.get("compound", 0))
    partition_df["neg_score"] = partition_df["sentiment"].apply(lambda x: x.get("neg", 0))
    partition_df["neu_score"] = partition_df["sentiment"].apply(lambda x: x.get("neu", 0))
    partition_df["pos_score"] = partition_df["sentiment"].apply(lambda x: x.get("pos", 0))

    return partition_df


def perform_sentiment_analysis(df, text_col="clean_text", parallel=True, n_workers=None, show_progress=True):
    """
    Applies VADER sentiment analysis on a text column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the text column
    text_col : str
        Column name containing text
    parallel : bool
        Whether to use multiprocessing
    n_workers : int, optional
        Number of worker processes (defaults to CPU cores - 1)
    show_progress : bool
        Whether to show progress bars
    """
    if df.empty:
        print("⚠️  Empty DataFrame — skipping sentiment analysis.")
        return df

    if not parallel:
        return compute_sentiment_scores_in_partition((df, text_col, show_progress))

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Split into chunks for multiprocessing
    df_split = np.array_split(df, n_workers)
    args = [(chunk, text_col, False) for chunk in df_split]  # Disable per-process tqdm

    results = []
    with Pool(processes=n_workers) as pool:
        if show_progress:
            for res in tqdm(pool.imap(compute_sentiment_scores_in_partition, args),
                            total=len(df_split),
                            desc="Processing chunks"):
                results.append(res)
        else:
            results = pool.map(compute_sentiment_scores_in_partition, args)

    return pd.concat(results, ignore_index=True)
