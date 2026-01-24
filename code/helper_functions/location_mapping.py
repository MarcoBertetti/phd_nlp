import pandas as pd

def clean_gadm_locations(gadm_gdf, admin_df):
    """Harmonize GADM administrative names to match FEWS/ACLED data conventions."""

    # --- Country-specific name normalizations -------------------------
    cameroon_mapping = {
        "sud": "south",
        "nord": "north",
        "est": "east",
        "ouest": "west",
        "extreme nord": "far north",
        "nord ouest": "northwest",
        "sud ouest": "southwest",
    }

    burundi_mapping = {
        "butaganzwa1": "butaganzwa",
        "butaganzwa2": "butaganzwa",
        "butaganzwal": "butaganzwa",
        "butaganzwa": "butaganzwa",
        "muramvya": "muramviya",
        "buhinyuza": "buhinvuza",
        "nyamurenza": "nvamurenza",
    }

    chad_mapping = {
        "barh el gazel": "barh el gazel nord",
    }

    # Apply lowercase-safe mapping
    def apply_map(df, country_col, name_col, country, mapping):
        mask = df[country_col].eq(country)
        df.loc[mask, name_col] = (
            df.loc[mask, name_col]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(mapping)
            .fillna(df.loc[mask, name_col])
        )

    apply_map(gadm_gdf, "NAME_0", "NAME_2", "burundi", burundi_mapping)
    apply_map(admin_df, "ADMIN0", "ADMIN2", "burundi", burundi_mapping)
    apply_map(gadm_gdf, "NAME_0", "NAME_1", "cameroon", cameroon_mapping)
    apply_map(gadm_gdf, "NAME_0", "NAME_2", "chad", chad_mapping)

    # --- Targeted fixes -----------------------------------------------
    gadm_gdf.loc[(gadm_gdf["NAME_2"] == "likuyani") & (gadm_gdf["NAME_1"] == "bungoma"), "NAME_1"] = "kakamega"
    gadm_gdf.loc[(gadm_gdf["NAME_1"] == "hodh ech chargui") & (gadm_gdf["NAME_2"] == "ouadane"), "NAME_1"] = "adrar"
    gadm_gdf.loc[(gadm_gdf["NAME_0"] == "sierra leone") & (gadm_gdf["NAME_1"] == "western"), "NAME_1"] = "western area"

    gadm_gdf.loc[(gadm_gdf["NAME_1"] == "al jawf") & (gadm_gdf["NAME_2"] == "az zahir"), "NAME_2"] = "az zahir al humayqan"
    admin_df.loc[(admin_df["ADMIN0"] == "yemen") & (admin_df["ADMIN1"] == "socotra"), "ADMIN1"] = "hadramaut"

    gadm_gdf.loc[gadm_gdf["NAME_1"] == "masvingo urban", "NAME_1"] = "masvingo"
    gadm_gdf.loc[
        (gadm_gdf["NAME_1"] == "harare") & (gadm_gdf["NAME_2"] == "harare"), "NAME_2"
    ] = "harare urban"

    # --- Malawi region aggregation -----------------------------------
    central_districts = [
        "dedza", "dowa", "kasungu", "lilongwe", "mchinji",
        "nkhotakota", "ntcheu", "ntchisi", "salima"
    ]
    northern_districts = ["chitipa", "karonga", "mzimba", "nkhata bay", "rumphi"]
    southern_districts = [
        "balaka", "blantyre", "chikwawa", "chiradzulu", "machinga",
        "mangochi", "mulanje", "mwanza", "neno", "nsanje",
        "phalombe", "thyolo", "zomba"
    ]

    mask = gadm_gdf["NAME_0"].eq("malawi")
    gadm_gdf.loc[mask & gadm_gdf["NAME_1"].isin(central_districts), "NAME_1"] = "central"
    gadm_gdf.loc[mask & gadm_gdf["NAME_1"].isin(northern_districts), "NAME_1"] = "northern"
    gadm_gdf.loc[mask & gadm_gdf["NAME_1"].isin(southern_districts), "NAME_1"] = "southern"

    gadm_gdf.loc[mask, "NAME_2"] = (
        gadm_gdf.loc[mask, "NAME_2"]
        .astype(str)
        .str.replace(" boma", "", regex=False)
        .str.replace(" city", "", regex=False)
        .str.replace(" town", "", regex=False)
    )

    # --- Uganda region aggregation -----------------------------------
    keep_names = {
        "buikwe", "bukomansimbi", "butambala", "buvuma", "gomba",
        "kalangala", "kalungu", "kyankwanzi", "luwero", "lwengo",
        "lyantonde", "mityana", "mpigi", "nakaseke", "amuria",
        "budaka", "bududa", "bukedea", "bukwo", "bulambuli", "butaleja",
        "buyende", "kaliro", "kibuku", "kumi", "kween", "luuka",
        "manafwa", "namayingo", "namutumba", "ngora", "serere", "abim",
        "agago", "alebtong", "amolatar", "amudat", "amuru", "dokolo",
        "kaabong", "koboko", "kole", "lamwo", "maracha", "napak",
        "nwoya", "otuke", "oyam", "zombo", "buhweju", "buliisa",
        "ibanda", "isingiro", "kibaale", "kiruhura", "kiryandongo",
        "kyegegwa", "mitooma", "ntoroko", "rubirizi", "sheema",
    }

    district_to_region = {  # (same mapping, unchanged)
        "adjumani": "northern", "amuria": "eastern", "apac": "northern", "arua": "northern",
        "budaka": "eastern", "bududa": "eastern", "bugiri": "eastern", "buikwe": "central",
        "bukedea": "eastern", "bukomansimbi": "central", "bukwo": "eastern", "bulambuli": "eastern",
        "bundibugyo": "western", "bushenyi": "western", "busia": "eastern", "butaleja": "eastern",
        "butambala": "central", "buvuma": "central", "buyende": "eastern", "gomba": "central",
        "gulu": "northern", "hoima": "western", "iganga": "eastern", "jinja": "eastern",
        "kabale": "western", "kabarole": "western", "kaberamaido": "eastern", "kalangala": "central",
        "kaliro": "eastern", "kalungu": "central", "kampala": "central", "kamuli": "eastern",
        "kamwenge": "western", "kanungu": "western", "kapchorwa": "eastern", "kasese": "western",
        "katakwi": "eastern", "kayunga": "central", "kiboga": "central", "kibuku": "eastern",
        "kisoro": "western", "kitgum": "northern", "kotido": "northern", "kumi": "eastern",
        "kween": "eastern", "kyankwanzi": "central", "kyenjojo": "western", "lira": "northern",
        "luuka": "eastern", "luwero": "central", "lwengo": "central", "lyantonde": "central",
        "manafwa": "eastern", "masaka": "central", "masindi": "western", "mayuge": "eastern",
        "mbale": "eastern", "mbarara": "western", "mityana": "central", "moroto": "northern",
        "moyo": "northern", "mpigi": "central", "mubende": "central", "mukono": "central",
        "nakapiripirit": "northern", "nakaseke": "central", "nakasongola": "central",
        "namayingo": "eastern", "namutumba": "eastern", "nebbi": "northern", "ngora": "eastern",
        "ntungamo": "western", "pader": "northern", "pallisa": "eastern", "rakai": "central",
        "rukungiri": "western", "sembabule": "central", "serere": "eastern", "sironko": "eastern",
        "soroti": "eastern", "tororo": "eastern", "wakiso": "central", "yumbe": "northern"
    }

    mask = gadm_gdf["NAME_0"] == "uganda"
    gadm_gdf.loc[mask & (~gadm_gdf["NAME_2"].isin(keep_names)), "NAME_2"] = gadm_gdf["NAME_1"]
    gadm_gdf.loc[mask, "NAME_1"] = (
        gadm_gdf.loc[mask, "NAME_1"].map(district_to_region).fillna(gadm_gdf.loc[mask, "NAME_1"])
    )

    return gadm_gdf, admin_df
