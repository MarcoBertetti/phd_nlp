import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import os
import requests
import zipfile
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging

# --------------------------- Configuration ---------------------------

event_codes = [
    # Conflict-related
    2, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 22, 42, 43,
    46, 47, 48, 56, 57, 74, 112, 121, 141, 162, 163, 171, 172, 173,

    # Agriculture, food, humanitarian
    70, 73, 74, 75,

    # Environmental (protest & policy)
    21, 22,

    # Economy and aid
    30, 31, 32, 33, 34, 35, 36, 37, 38,

    # Governance (optional — for early signals)
    100, 101, 102, 104,

    # Migration/refugees
    71, 72, 122
]

columns = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code",
    "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code",
    "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code", "Actor2Type1Code",
    "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode", "EventBaseCode",
    "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
    "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_Lat",
    "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

# ISO country codes for filtering
country_codes = [
    "BF", "BI", "CG", "CM", "ET", "GN", "KE", "LR", "MG", "ML", "MR",
    "MW", "MZ", "NG", "NE", "SD", "SL", "SO", "SS", "CD", "UG", "YE", "ZM", "ZW", "CF"
]

# FIPS codes for initial filtering
fips_codes = [
    "UV", "BY", "CG", "CM", "ET", "GV", "KE", "LI", "MA", "ML", "MR",
    "MI", "MZ", "NG", "NI", "SU", "SL", "SO", "OD", "CD", "UG", "YM", "ZA", "ZI", "CT"
]

fips_to_country = dict(zip(fips_codes, country_codes))
base_url = "http://data.gdeltproject.org/events/"

# --------------------------- Logging Setup ---------------------------

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("processing.log"), logging.StreamHandler()]
    )

# --------------------------- Helpers ---------------------------

def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)

def process_date(date_str, save_directory):
    """Download, extract, and filter one day of GDELT data."""
    file_url = f"{base_url}{date_str}.export.CSV.zip"
    file_path = os.path.join(save_directory, f"{date_str}.export.CSV.zip")
    csv_path = os.path.join(save_directory, f"{date_str}.export.CSV")

    if os.path.exists(csv_path):
        logging.info(f"File already exists, skipping: {csv_path}")
        return

    try:
        response = requests.get(file_url, timeout=(10, 30))
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded: {file_path}")

            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(save_directory)
            os.remove(file_path)

            # Read and assign GDELT schema
            df = pd.read_csv(csv_path, delimiter="\t", header=None, names=columns, engine="python")
            df.dropna(how="all", inplace=True)

            # Basic cleaning and filtering
            df["EventCode"] = pd.to_numeric(df["EventCode"], errors="coerce")
            df = df[df["ActionGeo_CountryCode"].isin(fips_codes)]
            df = df[df["EventCode"].isin(event_codes)]

            # Coordinate sanity filter
            if "ActionGeo_Lat" in df.columns and "ActionGeo_Long" in df.columns:
                df = df[
                    df["ActionGeo_Lat"].between(-90, 90) &
                    df["ActionGeo_Long"].between(-180, 180)
                ]

            # Save the cleaned file with header
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved filtered file: {csv_path}")

        else:
            logging.warning(f"Failed to download {file_url} (status {response.status_code})")

    except Exception as e:
        logging.warning(f"⚠️ Error processing {date_str}: {e}")

def download_gdelt_data(start_date, end_date, save_directory, max_workers=5):
    """Download and filter daily GDELT event files in parallel."""
    ensure_directory(save_directory)
    date_strings = [
        (start_date + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end_date - start_date).days + 1)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda date: process_date(date, save_directory), date_strings)

def consolidate_files(csv_directory, output_path):
    """Combine all filtered daily CSVs into one unified parquet dataset."""
    csv_files = [f for f in glob.glob(os.path.join(csv_directory, "*export.CSV")) if f.endswith("export.CSV")]
    if not csv_files:
        logging.warning("No CSV files found for consolidation.")
        return pd.DataFrame()

    df_list = []
    for csv in csv_files:
        try:
            # Your filtered files *already include headers* — so read normally
            df = pd.read_csv(csv, header=0)
            
            # Fallback: if a file somehow lost its header
            if "ActionGeo_CountryCode" not in df.columns:
                df = pd.read_csv(csv, delimiter="\t", header=None, names=columns, engine="python")
            
            df_list.append(df)
        except Exception as e:
            logging.warning(f"⚠️ Skipping malformed file {csv}: {e}")

    if not df_list:
        logging.error("❌ No valid CSV files to merge.")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    pre_filter_count = len(combined_df)

    # --- Replace FIPS with ISO ---
    combined_df["ActionGeo_CountryCode"] = combined_df["ActionGeo_CountryCode"].replace(fips_to_country)

    # --- Convert year to numeric for filtering ---
    combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce")

    # --- Apply country/year filters ---
    combined_df = combined_df[
        combined_df["ActionGeo_CountryCode"].isin(country_codes) &
        combined_df["Year"].between(2016, 2024)
    ]

    # --- Drop duplicates by event ID ---
    if "GLOBALEVENTID" in combined_df.columns:
        combined_df.drop_duplicates(subset=["GLOBALEVENTID"], inplace=True)

    post_filter_count = len(combined_df)
    logging.info(
        f"Rows before filters: {pre_filter_count:,} → after filters: {post_filter_count:,} "
        f"(dropped {pre_filter_count - post_filter_count:,})"
    )

    ensure_directory(os.path.dirname(output_path))
    combined_df.to_parquet(output_path, index=False)
    logging.info(f"✅ Saved consolidated data to: {output_path}")

    return combined_df

# --------------------------- Main Function ---------------------------

def main(save_directory="../data/gdelt/raw", output_file="../data/gdelt/consolidated/combined_data.parquet"):
    configure_logging()
    start_date = datetime(2020, 8, 20)
    end_date = datetime(2024, 10, 31)

    download_gdelt_data(start_date, end_date, save_directory)
    consolidate_files(save_directory, output_file)

if __name__ == "__main__":
    main()
