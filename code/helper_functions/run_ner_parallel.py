import spacy
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import re

def _normalize_location(name: str) -> str:
    """Lowercase + collapse whitespace; used for matching."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name

def run_ner_parallel(
    df,
    text_col: str = "clean_text",
    fews_countries=None,
    n_process: int = None,
    batch_size: int = 200,
):
    """
    Extracts up to three location entities (GPE/LOC) from text in parallel using spaCy.

    NER_admin0 selection logic (for FEWSNET work):

    1. Collect all GPE/LOC entities in document order.
    2. Among them, find entities that match FEWS countries (by normalized name).
       - Choose the FEWS country with the highest count in the article.
       - If tie: pick the one whose FIRST occurrence is earliest in the text.
    3. If no FEWS country is found:
       - Fallback to the first GPE/LOC entity (old behaviour).
    4. NER_admin1/2 are filled with the next distinct locations in document order
       (after removing the chosen admin0 index), if available.

    All NER_admin* outputs are lowercase strings.
    """

    # ---------- processes ----------
    if n_process is None:
        n_process = max(1, mp.cpu_count() - 1)

    # ---------- FEWS country normalization ----------
    if fews_countries is not None:
        # assume FEWS countries may already be lowercase, but normalize anyway
        fews_countries = list(fews_countries)
        fews_norm_to_canon = {
            _normalize_location(c): _normalize_location(c) for c in fews_countries
        }
        fews_set = set(fews_norm_to_canon.keys())
    else:
        fews_norm_to_canon = {}
        fews_set = None

    print(f"🔍 Running NER extraction on {len(df)} texts with {n_process} processes...")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
    nlp.max_length = 2_000_000

    texts = df[text_col].astype(str).tolist()

    admin0_list, admin1_list, admin2_list = [], [], []

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
        total=len(texts),
        desc="Running NER in parallel (FEWS-aware)"
    ):
        # 1) collect places in document order
        places = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
        norm_places = [_normalize_location(p) for p in places]

        # default outputs
        a0 = a1 = a2 = None

        if places:
            chosen_idx = None

            # 2) FEWS-aware selection for admin0
            if fews_set:
                # indices where a FEWS country appears
                fews_indices = [
                    i for i, np in enumerate(norm_places) if np in fews_set
                ]

                if fews_indices:
                    # Count FEWS mentions
                    from collections import Counter
                    fews_counts = Counter(
                        np for np in norm_places if np in fews_set
                    )
                    max_count = max(fews_counts.values())
                    # candidates with max count
                    best_norms = [
                        c for c, cnt in fews_counts.items() if cnt == max_count
                    ]

                    # tie-break by earliest position in text
                    best_norm = None
                    best_pos = 10**9
                    for cand in best_norms:
                        pos = norm_places.index(cand)
                        if pos < best_pos:
                            best_pos = pos
                            best_norm = cand

                    chosen_idx = best_pos
                    # map back to canonical FEWS spelling (lowercased)
                    a0 = fews_norm_to_canon.get(best_norm, best_norm)

            # 3) fallback: first GPE/LOC if no FEWS country matched
            if a0 is None:
                chosen_idx = 0
                a0 = norm_places[0]  # keep it lowercased

            # 4) admin1/admin2: next distinct locations in doc order
            remaining_indices = [
                i for i in range(len(places)) if i != chosen_idx
            ]
            if remaining_indices:
                a1 = norm_places[remaining_indices[0]]
            if len(remaining_indices) > 1:
                a2 = norm_places[remaining_indices[1]]

        admin0_list.append(a0)
        admin1_list.append(a1)
        admin2_list.append(a2)

    df["NER_admin0"] = admin0_list
    df["NER_admin1"] = admin1_list
    df["NER_admin2"] = admin2_list

    print("✅ NER extraction complete (FEWS-aware).")
    return df

# ---------------------------------------------------------
# Full Demonym Mapping for ALL African Countries → lowercase country name
# ---------------------------------------------------------

AFRICAN_DEMONYMS = {
    # North Africa
    "algerian": "algeria",
    "egyptian": "egypt",
    "libyan": "libya",
    "moroccan": "morocco",
    "tunisian": "tunisia",
    "sudanese": "sudan",
    "south sudanese": "south sudan",

    # West Africa
    "beninese": "benin",
    "burkinabe": "burkina faso",
    "cape verdean": "cabo verde",
    "ivorian": "côte d'ivoire",
    "ivorians": "côte d'ivoire",
    "gambian": "gambia",
    "ghanaian": "ghana",
    "guinean": "guinea",
    "bissau-guinean": "guinea-bissau",
    "liberian": "liberia",
    "malian": "mali",
    "mauritanian": "mauritania",
    "nigerien": "niger",
    "nigerian": "nigeria",
    "senegalese": "senegal",
    "sierran": "sierra leone",
    "togolese": "togo",

    # Central Africa
    "cameroonian": "cameroon",
    "central african": "central african republic",
    "chadian": "chad",
    "congolese": "democratic republic of the congo",  # default to DRC
    "equatoguinean": "equatorial guinea",
    "gabonese": "gabon",
    "sao tomean": "sao tome and principe",

    # East Africa
    "burundian": "burundi",
    "comorian": "comoros",
    "djiboutian": "djibouti",
    "eritrean": "eritrea",
    "ethiopian": "ethiopia",
    "kenyan": "kenya",
    "rwandan": "rwanda",
    "somali": "somalia",
    "somalian": "somalia",
    "somalians": "somalia",
    "south african": "south africa",
    "tanzanian": "tanzania",
    "ugandan": "uganda",

    # Southern Africa
    "angolan": "angola",
    "botswanan": "botswana",
    "basotho": "lesotho",
    "malawian": "malawi",
    "mauritian": "mauritius",
    "mozambican": "mozambique",
    "namibian": "namibia",
    "seychellois": "seychelles",
    "swazi": "eswatini",
    "eswatinian": "eswatini",
    "zambian": "zambia",
    "zimbabwean": "zimbabwe",
}

# ---------------------------------------------------------
# Function to rewrite clean_text by injecting the country name
# ---------------------------------------------------------

def inject_countries_from_demonyms(text: str) -> str:
    """
    Replace African demonyms with their country equivalent appended,
    so NER can detect countries that were only mentioned via demonym.

    Example:
        "73 die somalian bomb attack" →
        "73 die somalian somalia bomb attack"
    """
    if not isinstance(text, str):
        return text

    lower = text.lower()
    additions = []

    for dem, country in AFRICAN_DEMONYMS.items():
        if dem in lower:
            additions.append(country)

    if additions:
        # append unique country names to the text
        extra = " " + " ".join(sorted(set(additions))) + " "
        return text + extra

    return text
