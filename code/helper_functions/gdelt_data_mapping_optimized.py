import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopandas.tools import sjoin_nearest
from tqdm import tqdm
from p_tqdm import p_map
import os
import glob
import gc
import pygeos
from unidecode import unidecode
import json
from .location_mapping import clean_gadm_locations
# Enable pygeos for faster spatial operations if available
gpd.options.use_pygeos = True

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import shapely

# Ensure we are using only Shapely (not PyGEOS) to avoid compatibility issues
gpd.options.use_pygeos = False

def normalize_text(text):
    """Convert text to lowercase, remove accents, strip spaces, and standardize punctuation."""
    if pd.isna(text):  # Handle NaN values
        return None
    return unidecode(text).lower().strip().replace("-", " ").replace("'", "").replace("`", "")

with open("./helper_functions/names_map.json", "r") as f:
    name_mapping = json.load(f)

def replace_names(df, name_mapping):
    def replace_value(value):
        """Replace the value if it matches a key in the mapping."""
        if pd.isna(value):  # Ignore NaN values
            return value
        return name_mapping.get(value, value)  # Only replace if it's a key in name_mapping

    # Apply replacement function to the specified columns
    for col in ["NAME_0", "NAME_1", "NAME_2"]:
        if col in df.columns:
            df[col] = df[col].apply(replace_value)
    
    return df

# Restore the global variable declaration
gadm_gdf = None

# Restore the original load_gadm_data function
def load_gadm_data(gadm_path, fews_df):
    """
    Load the GADM dataset, ensuring valid and correct geometries before processing.
    """
    global gadm_gdf
    # Load the dataset
    gadm_gdf = gpd.read_file(gadm_path, layer="filtered_gadm_410", engine="pyogrio")

    # Ensure 'geometry' column exists
    if 'geometry' not in gadm_gdf.columns:
        raise ValueError("GADM dataset missing 'geometry' column!")

    # Drop missing geometries
    gadm_gdf = gadm_gdf.dropna(subset=['geometry'])

    # Convert MultiPolygons to Polygons (take the largest Polygon if needed)
    def ensure_polygon(geom):
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda g: g.area)  # Take the largest polygon
        return geom

    gadm_gdf['geometry'] = gadm_gdf['geometry'].apply(ensure_polygon)

    # Validate all geometries (fix invalid ones)
    gadm_gdf['geometry'] = gadm_gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

    # Ensure no null geometries exist
    gadm_gdf = gadm_gdf.dropna(subset=['geometry'])

    for col in ['NAME_0', 'NAME_1', 'NAME_2']:
        gadm_gdf[col] = gadm_gdf[col].astype(str).apply(normalize_text)

    for col in ['ADMIN0', 'ADMIN1', 'ADMIN2']:
        fews_df[col] = fews_df[col].astype(str).apply(normalize_text)

    gadm_gdf, fews_df = clean_gadm_locations(gadm_gdf, fews_df)
    gadm_gdf = replace_names(gadm_gdf, name_mapping)

    # Ensure expected columns exist
    expected_columns = ["NAME_1", "NAME_2", "geometry"]
    missing_columns = [col for col in expected_columns if col not in gadm_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in GADM data: {missing_columns}")

    # Filter to predefined locations
    predefined_locations = set(normalize_text(loc) for loc in fews_df['ADMIN0'].unique())
    gadm_gdf["NAME_0"] = gadm_gdf["NAME_0"].astype(str).apply(normalize_text)
    gadm_gdf = gadm_gdf[gadm_gdf["NAME_0"].isin(predefined_locations)]

    # Merge the gadm_gdf with the mapping_df to replace values
    mapping_df = pd.read_csv('/Users/marco.bertetti/Desktop/git_repos/phd_nlp/code/gadm_fewsnet_mapping.csv')
    gadm_gdf = gadm_gdf.merge(mapping_df, how='left', left_on=['NAME_0', 'NAME_1', 'NAME_2'], right_on=['country', 'admin1', 'admin2'])

    # Replace the original columns with matched columns
    gadm_gdf['NAME_0'] = gadm_gdf['ADMIN0_matched'].fillna(gadm_gdf['NAME_0'])
    gadm_gdf['NAME_1'] = gadm_gdf['ADMIN1_matched'].fillna(gadm_gdf['NAME_1'])
    gadm_gdf['NAME_2'] = gadm_gdf['ADMIN2_matched'].fillna(gadm_gdf['NAME_2'])

    # Drop the matched columns
    gadm_gdf.drop(columns=['country', 'admin1', 'admin2', 'ADMIN0_matched', 'ADMIN1_matched', 'ADMIN2_matched'], inplace=True)

    # Ensure correct CRS
    gadm_gdf = gadm_gdf.to_crs(epsg=4326)
    
    # Try building the spatial index
    try:
        gadm_gdf.sindex  # Force creation of spatial index
    except Exception as e:
        raise ValueError(f"Failed to create spatial index: {e}")
    
    return gadm_gdf, fews_df

def get_admin_areas_batch(lat_list, lon_list):
    # Filter out invalid coordinates
    valid_coords = [
        (lat, lon)
        for lat, lon in zip(lat_list, lon_list)
        if pd.notna(lat) and pd.notna(lon)
        and -90 <= lat <= 90 and -180 <= lon <= 180
    ]
    
    if not valid_coords:
        return [None] * len(lat_list), [None] * len(lat_list)

    lats, lons = zip(*valid_coords)

    try:
        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in zip(lats, lons)],
            crs="EPSG:4326"
        )
        projected_crs = "EPSG:3857"
        points_gdf = points_gdf.to_crs(projected_crs)
        gadm_projected = gadm_gdf.to_crs(projected_crs)
        match = sjoin_nearest(points_gdf, gadm_projected, how="left", distance_col="distance")
        admin1_list = match['NAME_1'].tolist() if 'NAME_1' in match.columns else [None] * len(lats)
        admin2_list = match['NAME_2'].tolist() if 'NAME_2' in match.columns else [None] * len(lats)
    except Exception as e:
        admin1_list = [None] * len(lat_list)
        admin2_list = [None] * len(lat_list)
        return admin1_list, admin2_list

    # Pad results back to original input length
    result_admin1, result_admin2 = [], []
    idx = 0
    for lat, lon in zip(lat_list, lon_list):
        if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
            result_admin1.append(admin1_list[idx])
            result_admin2.append(admin2_list[idx])
            idx += 1
        else:
            result_admin1.append(None)
            result_admin2.append(None)

    return result_admin1, result_admin2

# Restore the original process_gdelt_data function
def process_gdelt_data(df, gadm_gdf, output_dir, num_cpus=8, batch_size=10000):
    """
    Process the GDELT data efficiently using batch processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    unique_monthyears = sorted(df['MonthYear'].unique())

    for monthyear in unique_monthyears:
        output_file = os.path.join(output_dir, f"mapped_gdelt_{monthyear}.parquet")
        if os.path.exists(output_file):
            print(f"Skipping {monthyear}, already processed.")
            continue

        print(f"Processing {monthyear}...")
        df_chunk = df[df['MonthYear'] == monthyear].copy()
        df_chunk.reset_index(drop=True, inplace=True)
        lat_list, lon_list = df_chunk['ActionGeo_Lat'].tolist(), df_chunk['ActionGeo_Long'].tolist()

        results_admin1, results_admin2 = [], []
        for i in tqdm(range(0, len(lat_list), batch_size), desc="Processing Batches"):
            batch_lat, batch_lon = lat_list[i:i+batch_size], lon_list[i:i+batch_size]
            batch_admin1, batch_admin2 = get_admin_areas_batch(batch_lat, batch_lon)

            # Ensure batch length consistency
            min_batch_length = min(len(batch_admin1), len(batch_lat))
            batch_admin1 = batch_admin1[:min_batch_length]
            batch_admin2 = batch_admin2[:min_batch_length]

            results_admin1.extend(batch_admin1)
            results_admin2.extend(batch_admin2)
        
        # Ensure final results match df_chunk
        min_length = min(len(results_admin1), len(df_chunk))
        results_admin1 = results_admin1[:min_length]
        results_admin2 = results_admin2[:min_length]
        df_chunk = df_chunk.iloc[:min_length]

        df_chunk['ADMIN1'], df_chunk['ADMIN2'] = results_admin1, results_admin2
        df_chunk.to_parquet(output_file, index=False)
        print(f"Saved {monthyear} to {output_file}")

        del df_chunk
        gc.collect()

    print("Merging all processed files...")
    parquet_files = glob.glob(os.path.join(output_dir, "mapped_gdelt_20*.parquet"))
    df_result = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    
    # Convert SQLDATE to numeric to avoid ArrowTypeError
    if 'SQLDATE' in df_result.columns:
        df_result['SQLDATE'] = pd.to_numeric(df_result['SQLDATE'], errors='coerce')
    
    final_output_file = os.path.join(output_dir, "mapped_gdelt_final.parquet")
    df_result.to_parquet(final_output_file, index=False)
    print(f"Final data saved to {final_output_file}")
    return df_result

def consolidate_and_merge_fews(mapped_gdelt_dir, fews_df, period=None):
    parquet_file = glob.glob(os.path.join(mapped_gdelt_dir, 'mapped_gdelt_final.parquet'))
    df = pd.read_parquet(parquet_file)

    df = df[['SQLDATE', 'MonthYear', 'EventCode', 'EventBaseCode', 'Actor1Geo_FullName',
             'ActionGeo_CountryCode', 'ADMIN1', 'ADMIN2', 'Actor1Geo_Lat', 'Actor1Geo_Long',
             'SOURCEURL', 'NumMentions', 'NumSources', 'NumArticles']]
    df.rename(columns={'MonthYear': 'period'}, inplace=True)

    fews_df['period'] = fews_df['period'].astype(str)
    fews_df[['ADMIN0','ADMIN1','ADMIN2']] = fews_df[['ADMIN0','ADMIN1','ADMIN2']].astype(str).apply(lambda x: x.str.lower())
    df[['ADMIN1','ADMIN2']] = df[['ADMIN1','ADMIN2']].astype(str).apply(lambda x: x.str.lower())
    df['period'] = df['period'].astype(str)

    merged = pd.merge(fews_df, df, on=['ADMIN1','ADMIN2','period'], how='outer', indicator=True).sort_values('SQLDATE')
    if period: merged = merged[merged['period'] == period]
    df = merged.groupby(['ADMIN0','ADMIN1','ADMIN2','CS_score','period'], dropna=False).agg({
        'SQLDATE': lambda x: list(x.dropna()),
        'EventCode': lambda x: list(x.dropna()),
        'SOURCEURL': lambda x: list(x.dropna()),
        'NumMentions': lambda x: list(x.dropna()),
        'NumSources': lambda x: list(x.dropna()),
        'NumArticles': lambda x: list(x.dropna())
    }).reset_index().sort_values(['ADMIN0','ADMIN1','ADMIN2','period', 'CS_score'])
    return df