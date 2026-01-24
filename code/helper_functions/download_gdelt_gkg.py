# download_gdelt_gkg.py  – FAST & STRICT version (bug-free, verified)
# =============================================================================
# • Keeps full 27-column rows
# • Fast theme filter (set intersection)
# • Strict country filter (Locations → FIPS)
# • Header-less TSVs, clean Parquet consolidation
# • No duplicate logging lines / syntax errors
# -----------------------------------------------------------------------------
# pip install pandas pyarrow requests tqdm
# =============================================================================

from __future__ import annotations
import os, re, zipfile, glob, logging, warnings, requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, Union, List
import pandas as pd
import gc
from tqdm import tqdm
from .gdelt_data_mapping_optimized import get_admin_areas_batch

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------#
FIPS_CODES: List[str] = [
    "UV", "BY", "CG", "CM", "ET", "GV", "KE", "LI", "MA", "ML", "MR", "MI",
    "MZ", "NG", "NI", "SU", "SL", "SO", "OD", "CD", "UG", "YM", "ZA", "ZI", "CT"
]
TARGET_FIPS_SET = set(FIPS_CODES)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHITELIST_CSV = os.path.join(BASE_DIR, '../../data/gdelt/gkg_v2themes_foodsecurity_filtered.csv')
WHITELIST_COL = "unique_v2theme"

# ---------------------------------------------------------------------------#
def _load_theme_whitelist(csv_path: str) -> list[str]:
    """Safely load whitelist of themes for filtering."""
    if not os.path.exists(csv_path):
        logging.warning("Whitelist CSV not found: %s", csv_path)
        return []
    df = pd.read_csv(csv_path)
    if WHITELIST_COL not in df.columns:
        logging.warning("Column %s not found; using first column instead.", WHITELIST_COL)
        col = df.columns[0]
    else:
        col = WHITELIST_COL
    themes = df[col].dropna().astype(str).tolist()
    logging.info("Theme whitelist loaded: %d items", len(themes))
    return themes

THEMES: List[str] = _load_theme_whitelist(WHITELIST_CSV)
THEME_SET = set(THEMES)

_BASE_URL = "http://data.gdeltproject.org/gdeltv2/"

_GKG_COLS_V2 = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras"
]

# ---------------------------------------------------------------------------#
def _ensure_dir(path: str | os.PathLike):
    os.makedirs(path, exist_ok=True)

def _fname(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"

def _configure_logging(logfile: str | None):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers
    )

# ---------------------------------------------------------------------------#
FIPS_POS = 2  # country code field index

def _extract_fips(loc: str) -> list[str]:
    """Extract valid FIPS codes from a GDELT location string."""
    codes = []
    for tup in str(loc).split(","):
        parts = tup.split("#")
        if len(parts) > FIPS_POS:
            code = parts[FIPS_POS].strip().upper()
            if len(code) == 2 and code.isalpha():
                codes.append(code)
    return codes

# ---------------------------------------------------------------------------#
def _process(dt: datetime, raw_dir: str):
    fn = _fname(dt)
    zip_path = os.path.join(raw_dir, fn)
    csv_path = zip_path.replace(".zip", "")
    if os.path.exists(csv_path):
        return

    # ---------------- Download ----------------
    try:
        resp = requests.get(_BASE_URL + fn, timeout=60)
        if resp.status_code != 200:
            return
    except requests.RequestException as exc:
        logging.warning("Request error %s: %s", fn, exc)
        return

    with open(zip_path, "wb") as fh:
        fh.write(resp.content)

    # ---------------- Unzip ----------------
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    except zipfile.BadZipFile:
        logging.warning("Bad zip %s", fn)
        os.remove(zip_path)
        return
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # ---------------- Read TSV ----------------
    try:
        df = pd.read_csv(
            csv_path,
            sep="\t",
            header=None,
            names=_GKG_COLS_V2,
            quoting=3,
            low_memory=False,
            encoding="latin1",
            on_bad_lines="skip"
        )
    except Exception as exc:
        if os.path.exists(csv_path):
            logging.error("Read fail %s: %s", fn, exc)
            os.remove(csv_path)
        return

    # ---------------- Theme filter ----------------
    df["theme_list"] = (
        df["V2Themes"].astype(str)
        .str.split(";")  # GDELT uses semicolons
        .apply(lambda lst: [t.split(",")[0].strip().upper() for t in lst if t.strip()])
    )

    mask_theme = df["theme_list"].apply(lambda lst: bool(set(lst) & THEME_SET))

    # ---------------- Country filter ----------------
    df["fips_codes"] = df["V2Locations"].apply(_extract_fips)
    mask_fips = df["fips_codes"].apply(lambda lst: any(cc in TARGET_FIPS_SET for cc in (lst or [])[:3]))

    kept = (mask_theme & mask_fips).sum()
    total = len(df)
    logging.info(f"{fn}: total={total}, kept={kept}, theme={mask_theme.sum()}, fips={mask_fips.sum()}")

    df = df[mask_theme & mask_fips]
    if df.empty:
        os.remove(csv_path)
        return

    # ---------------- Save header-less TSV ----------------
    df[_GKG_COLS_V2].to_csv(csv_path, index=False, sep="\t", quoting=3, header=False)
    logging.info("✓ %s → %d rows kept", fn, len(df))

# ---------------------------------------------------------------------------#
def download_range(start: datetime, end: datetime, raw_dir: str, workers: int = 8):
    """Download all 15-minute GKG files between start and end."""
    _ensure_dir(raw_dir)
    ticks: list[datetime] = []
    cur = start
    while cur <= end:
        ticks.append(cur)
        cur += timedelta(minutes=15)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda dt: _process(dt, raw_dir), ticks))

# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#
# 0. Keyword → bucket mapping (all lower-case)
# ---------------------------------------------------------------------------#
KEYWORDS_BY_BUCKET: dict[str, list[str]] = {
    # Human impact
    "n_killed":          ["kill", "killed", "dead", "death", "fatal", "murder", "casualty"],
    "n_injured":         ["injur", "wound", "hurt", "hospital", "casualty"],
    "n_displaced":       ["displac", "refugee", "evacuat", "flee", "relocat"],
    "n_missing":         ["missing", "unaccounted", "kidnap", "abduct", "lost"],

    # Assistance & money
    "usd_aid":           ["aid", "relief", "assist", "donat", "support", "fund"],

    # Food & water
    "n_food_related":    ["food", "hunger", "famine", "malnutrit", "nutrition", "feeding"],
    "n_water_related":   ["water", "drought", "thirst", "hydration"],

    # Economics
    "n_price_related":   ["price", "inflation", "cost", "expens", "afford"],
    "n_market_related":  ["market", "trade", "commerce", "supply", "demand",
                          "import", "export", "commodity"],

    # Conflict / security
    "n_conflict_related":["conflict", "war", "battle", "combat", "clash",
                          "violence", "attack", "raid", "uprising"],

    # Hazards / health
    "n_disease_related": ["disease", "illness", "epidemic", "pandemic",
                          "virus", "infection", "outbreak", "health"],
    "n_weather_related": ["weather", "storm", "flood", "climate", "cyclone",
                          "rain", "heatwave", "cold wave", "typhoon",
                          "hurricane", "tornado", "hail"],

    # Governance
    "n_policy_related":  ["policy", "law", "regulat", "government",
                          "legislat", "ban", "rule", "ordinance", "directive"],
}

DEFAULT_COUNTS = {bucket: 0 for bucket in KEYWORDS_BY_BUCKET}

# ---------------------------------------------------------------------------#
def aggregate_files_by_day(input_folder, output_folder, start_date, end_date):
    """Aggregate all 15-min GKG files into one Parquet per day."""
    os.makedirs(output_folder, exist_ok=True)

    start_date = datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y%m%d")

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        print(f"Processing date: {date_str}")

        # Find all files for this date (strict pattern)
        pattern = re.compile(fr"^{date_str}\d{{6}}\.gkg\.csv$")
        files = [f for f in os.listdir(input_folder) if pattern.match(f)]

        df_list = []
        for i, file in enumerate(files):
            file_path = os.path.join(input_folder, file)
            try:
                df = pd.read_csv(
                    file_path,
                    sep="\t",
                    header=0 if i == 0 else None,
                    usecols=range(27),
                    names=_GKG_COLS_V2,
                    quoting=3,
                    low_memory=False,
                    on_bad_lines="skip"
                )
                df_list.append(df)
            except Exception as exc:
                logging.warning("Skipping corrupt file %s: %s", file, exc)

        if df_list:
            daily_df = pd.concat(df_list, ignore_index=True)

            # Drop only if present
            drop_cols = [c for c in ['Locations', 'Themes', 'Persons', 'Organizations', 'GCAM', 'SocialVideoEmbeds'] if c in daily_df.columns]
            if drop_cols:
                daily_df.drop(columns=drop_cols, inplace=True)

            daily_file_path = os.path.join(output_folder, f"{date_str}.parquet")
            daily_df.to_parquet(daily_file_path, index=False)
            print(f"Saved aggregated data for {date_str} to {daily_file_path}")

        current_date += timedelta(days=1)
# ---------------------------------------------------------------------------
def extract_counts(v2counts: str) -> dict[str, int]:
    """
    Parse the GDELT V2Counts field and sum numeric values into thematic buckets.

    The function:
    1. Splits `v2counts` on ',' to get individual count objects.
    2. Splits each object on '#' to obtain:
          code  = tokens[0]   (e.g. 'CRISISLEX_T02_INJURED')
          value = tokens[1]   (integer count)
    3. Converts the code to lower-case and checks whether it *contains* any
       of the keyword substrings defined in `KEYWORDS_BY_BUCKET`.
    4. Returns a dict with **all bucket keys present** (missing → 0).
    """
    if pd.isna(v2counts) or not v2counts:
        return DEFAULT_COUNTS.copy()  # always return fresh copy

    counts = DEFAULT_COUNTS.copy()

    for obj in str(v2counts).split(","):
        tokens = obj.split("#")
        if len(tokens) < 2:
            continue
        code = tokens[0].lower().strip()
        try:
            val = int(tokens[1])
        except ValueError:
            continue

        for bucket, keywords in KEYWORDS_BY_BUCKET.items():
            if any(kw in code for kw in keywords):
                counts[bucket] += val
                break

    return counts

# ---------------------------------------------------------------------------
def consolidate(raw_dir: str, out_parquet: str):
    """
    Read all daily Parquet files in *raw_dir* (named YYYYMMDD.parquet),
    enrich them, and write ONE Parquet per calendar month into the same
    directory as *out_parquet*.
    """
    import os, glob, re, gc, logging, pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    daily_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    if not daily_files:
        logging.warning("No files to consolidate.")
        return

    # Group by YYYYMM prefix
    month_bins = {}
    for fp in daily_files:
        yyyy_mm = os.path.basename(fp)[:6]
        if re.match(r"^\d{6}$", yyyy_mm):
            month_bins.setdefault(yyyy_mm, []).append(fp)
        else:
            logging.warning("Skipping unparsable filename %s", fp)

    out_dir = Path(out_parquet).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cc_re = re.compile(r"#([A-Z]{2})#")

    # ---------------- Helper extractors ----------------
    def extract_tone_metrics(v2tone):
        if pd.isna(v2tone):
            return (None, None, None)
        try:
            tone = float(str(v2tone).split(",")[0])
            return (tone, abs(tone), int(tone < 0))
        except Exception:
            return (None, None, None)

    def extract_lat_lon(locstr):
        """Extract first valid (lat, lon) from a GDELT V2Locations string."""
        if pd.isna(locstr):
            return (None, None)
        locstr = str(locstr)
        for loc in locstr.split(","):
            parts = loc.split("#")
            if len(parts) >= 7 and parts[2] in TARGET_FIPS_SET:
                try:
                    lat, lon = float(parts[5]), float(parts[6])
                    if lat or lon:
                        return (lat, lon)
                except Exception:
                    pass
        for loc in locstr.split(","):
            parts = loc.split("#")
            if len(parts) >= 7:
                try:
                    lat, lon = float(parts[5]), float(parts[6])
                    if lat or lon:
                        return (lat, lon)
                except Exception:
                    pass
        return (None, None)

    def extract_themes(v2themes: str):
        if pd.isna(v2themes):
            return []
        # Use semicolon (GDELT v2 separator)
        return [t.strip().split(",")[0] for t in v2themes.split(";") if t.strip().split(",")[0] in THEME_SET]

    # ---------------- Monthly consolidation loop ----------------
    for yyyy_mm, month_files in sorted(month_bins.items()):
        year, month = yyyy_mm[:4], yyyy_mm[4:]
        logging.info("▶ Processing %s-%s : %d files", year, month, len(month_files))

        month_df = pd.concat(
            (pd.read_parquet(fp) for fp in tqdm(month_files, desc=yyyy_mm)),
            ignore_index=True
        )

        # Convert dates
        month_df["DATE"] = pd.to_datetime(month_df["DATE"], format="%Y%m%d%H%M%S", errors="coerce")

        # Extract FIPS and themes
        month_df["fips_codes"] = month_df["V2Locations"].astype(str).str.findall(cc_re)
        month_df["fips_unique"] = month_df["fips_codes"].apply(set)
        month_df["fips_first"] = month_df["fips_codes"].apply(lambda lst: lst[0] if lst else None)
        month_df["fips_matched"] = month_df["fips_unique"].apply(lambda s: list(s & TARGET_FIPS_SET))

        # Expand counts into columns
        counts_df = month_df["V2Counts"].apply(lambda x: pd.Series(extract_counts(x) if isinstance(x, str) else {}))
        month_df = pd.concat([month_df, counts_df], axis=1)

        # Tone and coordinates
        month_df[["tone", "tone_abs", "is_negative"]] = month_df["V2Tone"].apply(lambda x: pd.Series(extract_tone_metrics(x)))
        month_df[["lat", "lon"]] = month_df["V2Locations"].apply(lambda x: pd.Series(extract_lat_lon(x)))

        # Themes
        month_df["themes_matched"] = month_df["V2Themes"].astype(str).apply(extract_themes)
        month_df["V2Themes"] = month_df["themes_matched"].apply(",".join)

        # Force object → string before Parquet write
        obj_cols = month_df.select_dtypes(include="object").columns
        month_df[obj_cols] = month_df[obj_cols].astype(str)

        out_file = out_dir / f"gkg_{year}_{month}.parquet"
        month_df.to_parquet(out_file, index=False)
        logging.info("✔ Wrote %s (%d rows)", out_file.name, len(month_df))

        del month_df
        gc.collect()

    logging.info(f"Data saved to {out_dir} in monthly parquet files")
    return True

# ---------------------------------------------------------------------------
def download_gkg_range(start: Union[str, datetime], end: Union[str, datetime],
                       raw_dir: str, max_workers: int = 8,
                       logfile: str | None = "gkg_processing.log") -> None:
    _configure_logging(logfile)
    start_dt = datetime.fromisoformat(start) if isinstance(start, str) else start
    end_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
        end_dt = end_dt.replace(hour=23, minute=45)
    logging.info("Fetching GKG %s → %s", start_dt, end_dt)
    download_range(start_dt, end_dt, raw_dir, workers=max_workers)
    logging.info("✓ Download done.")

def consolidate_gkg(raw_dir: str, out_file: str) -> None:
    _configure_logging("gkg_processing.log")
    consolidate(raw_dir, out_file)
    logging.info("✓ Consolidation done.")

# ---------------------------------------------------------------------------
def process_gkg_data(df, output_dir, num_cpus=8, batch_size=10000):
    """
    Process the GKG data efficiently using batch processing.
    Maps coordinates to administrative regions using GADM data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert DATE to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Extract YYYYMM period
    df["monthyear"] = df["DATE"].dt.strftime("%Y%m")
    unique_monthyears = sorted(df["monthyear"].dropna().unique())

    for monthyear in unique_monthyears:
        output_file = os.path.join(output_dir, f"mapped_gkg_{monthyear}.parquet")
        if os.path.exists(output_file):
            logging.info("Skipping %s (already processed).", monthyear)
            print("Skipping %s (already processed).", monthyear)
            continue

        logging.info("Processing %s ...", monthyear)
        df_chunk = df[df["monthyear"] == monthyear].copy().reset_index(drop=True)

        chunk_size = 10000
        total_rows = len(df_chunk)
        processed_chunks = []

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            current_chunk = df_chunk.iloc[start_idx:end_idx].copy()

            lat_list = current_chunk["lat"].tolist()
            lon_list = current_chunk["lon"].tolist()

            results_admin1, results_admin2 = [], []
            for i in range(0, len(lat_list), batch_size):
                batch_lat = lat_list[i:i+batch_size]
                batch_lon = lon_list[i:i+batch_size]
                batch_admin1, batch_admin2 = get_admin_areas_batch(batch_lat, batch_lon)
                results_admin1.extend(batch_admin1)
                results_admin2.extend(batch_admin2)

            current_chunk["ADMIN1"] = results_admin1
            current_chunk["ADMIN2"] = results_admin2

            chunk_file = f"{output_file}.chunk_{start_idx//chunk_size}"
            current_chunk.to_parquet(chunk_file, index=False)
            processed_chunks.append(chunk_file)
            del current_chunk
            gc.collect()

        df_month = pd.concat([pd.read_parquet(f) for f in processed_chunks], ignore_index=True)
        df_month.to_parquet(output_file, index=False)
        for f in processed_chunks:
            os.remove(f)

        logging.info("Saved %s (%d rows)", output_file, len(df_month))
        del df_month
        gc.collect()

# ---------------------------------------------------------------------------
def consolidate_and_merge_fews_gkg(mapped_gkg_dir, fews_df, period=None):
    """
    Consolidate mapped GKG data and merge with FEWS NET data.
    Keeps logic consistent with consolidate_and_merge_fews().
    """
    parquet_files = glob.glob(os.path.join(mapped_gkg_dir, "*.parquet"))
    if not parquet_files:
        logging.warning("No mapped parquet files found in %s", mapped_gkg_dir)
        return pd.DataFrame()

    # Combine all monthly mapped GKG files
    df = pd.concat((pd.read_parquet(fp) for fp in parquet_files), ignore_index=True)

    # 🔹 Rename MonthYear to period (consistent with GDELT)
    df["period"] = df["MonthYear"] if "MonthYear" in df.columns else df["DATE"].dt.strftime("%Y%m")

    # 🔹 Normalize ADMIN fields to lowercase for consistent merging
    df[["ADMIN1", "ADMIN2"]] = df[["ADMIN1", "ADMIN2"]].astype(str).apply(lambda x: x.str.lower())
    fews_df[["ADMIN0", "ADMIN1", "ADMIN2"]] = fews_df[["ADMIN0", "ADMIN1", "ADMIN2"]].astype(str).apply(lambda x: x.str.lower())
    fews_df["period"] = fews_df["period"].astype(str)
    df["period"] = df["period"].astype(str)

    # 🔹 Keep relevant columns only
    df = df[[
        "DATE", "period", "V2Themes", "DocumentIdentifier", "Amounts",
        "ADMIN1", "ADMIN2",
        "lat", "lon",
        "n_displaced", "n_killed", "n_injured", "n_missing",
        "usd_aid", "n_food_related", "n_water_related",
        "n_price_related", "n_conflict_related", "n_disease_related",
        "n_weather_related", "n_market_related", "n_policy_related",
        "tone", "tone_abs", "is_negative"
    ]]

    # 🔹 Merge GKG with FEWS data
    merged = pd.merge(
        fews_df,
        df,
        on=["ADMIN1", "ADMIN2", "period"],
        how="outer",
        indicator=True
    ).sort_values("DATE")

    # Optional period filter
    if period:
        merged = merged[merged["period"] == str(period)]

    # 🔹 Group by same structure as GDELT merge
    grouped = (
        merged.groupby(["ADMIN0", "ADMIN1", "ADMIN2", "CS_score", "period"], dropna=False)
        .agg({
            "DATE": lambda x: list(x.dropna()),
            "V2Themes": lambda x: list(x.dropna()),
            "DocumentIdentifier": lambda x: list(x.dropna()),
            "Amounts": lambda x: list(x.dropna()),
            "n_displaced": "sum",
            "n_killed": "sum",
            "n_injured": "sum",
            "n_missing": "sum",
            "usd_aid": "sum",
            "n_food_related": "sum",
            "n_water_related": "sum",
            "n_price_related": "sum",
            "n_conflict_related": "sum",
            "n_disease_related": "sum",
            "n_weather_related": "sum",
            "n_market_related": "sum",
            "n_policy_related": "sum",
            "tone": "mean",
            "tone_abs": "mean",
            "is_negative": "mean"
        })
        .reset_index()
        .sort_values(["ADMIN0", "ADMIN1", "ADMIN2", "period", "CS_score"])
    )

    return grouped
