###############################################################################
# 7) Re-aggregate articles into arrays
###############################################################################
from itertools import chain
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Category dictionary (same as used upstream)
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
        "extreme weather", "heatwave", "cold snap", "hurricane",
        "cyclone", "typhoon", "wildfire"
    ],
    "food_insecurity": [
        "food insecurity", "food shortage", "malnutrition", "starvation",
        "undernourished", "lack of food", "food crisis"
    ]
}

# ---------------------------------------------------------------------------
# Re-aggregation function
# ---------------------------------------------------------------------------
def aggregate_articles(df, group_cols, original_list_cols, original_df):
    """
    Group by the identifying columns (e.g., ADMIN0, ADMIN1, ADMIN2, period)
    and aggregate both processed columns and original arrays.

    Args:
        df : DataFrame
            Processed DataFrame with extracted features
        group_cols : list
            Columns to group by (e.g., ['ADMIN0', 'ADMIN1', 'ADMIN2', 'period'])
        original_list_cols : list
            Columns from the original file to merge back (arrays of SOURCEURL, etc.)
        original_df : DataFrame
            Original expanded dataset with list columns

    Returns:
        grouped : aggregated DataFrame
    """
    merge_cols = [col for col in group_cols if col != "original_idx"]

    # --- Aggregation rules ---------------------------------------------------
    agg_dict = {
        "text": list,
        "clean_text": list,
        "sentiment": list,
        "compound_score": list,
        "topic": list,
        "topic_probability": list,
        "topic_label": list,
        "severity": list,
        "named_entities": list,
        "pred_impact_type": list,
        "pred_urgency": list,
    }

    # Add FS-specific metrics dynamically
    for cat in FS_CATEGORIES:
        pred_resource_col = f"pred_resource_{cat}"
        cat_count_col = f"{cat}_count"
        cat_freq_col = f"{cat}_freq"
        cat_sent_col = f"{cat}_sentences"

        agg_dict[cat_count_col] = "sum"
        agg_dict[cat_freq_col] = "sum"
        agg_dict[pred_resource_col] = "sum"
        # Flatten nested lists safely
        agg_dict[cat_sent_col] = (
            lambda x: list(chain.from_iterable(i for i in x if isinstance(i, (list, np.ndarray))))
            if cat_sent_col in df.columns else (lambda _: [])
        )

    # Keep only existing columns
    existing_cols = df.columns.intersection(agg_dict.keys())
    final_agg = {col: agg_dict[col] for col in existing_cols}

    # --- Aggregate -----------------------------------------------------------
    grouped = df.groupby(merge_cols, dropna=False).agg(final_agg).reset_index()

    # --- Merge with original data -------------------------------------------
    grouped = grouped.merge(
        original_df[merge_cols + original_list_cols],
        on=merge_cols,
        how="left"
    )

    # --- Safe summation helpers ---------------------------------------------
    def safe_sum(arr):
        """Sum numeric arrays while ignoring None / NaN values."""
        if isinstance(arr, (np.ndarray, list)):
            valid = [v for v in arr if isinstance(v, (int, float)) and not np.isnan(v)]
            return int(np.sum(valid)) if valid else 0
        return 0

    for col, new_col in [
        ("NumMentions", "TotalNumMentions"),
        ("NumSources", "TotalNumSources"),
        ("NumArticles", "TotalNumArticles")
    ]:
        grouped[new_col] = grouped[col].apply(safe_sum) if col in grouped.columns else 0

    return grouped
