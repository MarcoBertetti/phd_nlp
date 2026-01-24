###############################################################################
# 5) Topic Modeling (BERTopic)
###############################################################################
import matplotlib.pyplot as plt
import seaborn as sns

def topic_modeling_bertopic(df, text_col='clean_text', language='english'):
    """Performs BERTopic modeling on the given dataframe's text column."""
    documents = df[text_col].tolist()
    num_docs = len(documents)

    if num_docs < 50:
        print(f"Skipping topic modeling. Only {num_docs} documents found.")
        df[['topic', 'topic_probability']] = -1, None
        return df, None

    import torch
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    num_threads = int(os.getenv("NSLOTS", os.cpu_count()))  # Use allocated slots
    torch.set_num_threads(num_threads)
    print(f"Setting torch to use {num_threads} threads.")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model only once
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=10, n_components=5, metric='cosine', random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=5, metric='euclidean', prediction_data=True),
        nr_topics="auto",
        language=language,
        calculate_probabilities=False
    )

    df['topic'], df['topic_probability'] = topic_model.fit_transform(documents)

    return df, topic_model

###############################################################################
# 6) Analyze Topics & Sentiment
###############################################################################
def analyze_topics_and_sentiment(df, topic_model, conflict_keywords, climate_keywords, food_security_keywords):
    """Categorizes topics based on keywords and analyzes sentiment severity."""

    topic_info = topic_model.get_topic_info()

    def get_topic_words(topic_id):
        return [word.lower() for word, _ in topic_model.get_topic(topic_id)] if topic_id != -1 else []

    def find_topics_by_keywords(keywords_list):
        return [
            topic_id for topic_id in topic_info['Topic'] if topic_id != -1 and 
            any(kw in w for w in get_topic_words(topic_id) for kw in keywords_list)
        ]

    # Preprocess keywords once
    conflict_keywords = {kw.lower().strip() for kw in conflict_keywords}
    climate_keywords = {kw.lower().strip() for kw in climate_keywords}
    food_security_keywords = {kw.lower().strip() for kw in food_security_keywords}

    # Find matching topics
    conflict_topics = find_topics_by_keywords(conflict_keywords)
    climate_topics = find_topics_by_keywords(climate_keywords)
    food_security_topics = find_topics_by_keywords(food_security_keywords)

    # Vectorized topic categorization
    topic_map = {t: 'Conflict' for t in conflict_topics} | \
                {t: 'Climate Change' for t in climate_topics} | \
                {t: 'Food Security' for t in food_security_topics}

    df['topic_label'] = df['topic'].map(topic_map).fillna('Other')
    df['severity'] = -df['compound_score']

    # Compute and display severity statistics
    severity_by_topic = df.groupby('topic_label', observed=True)['severity'].mean().reset_index()
    print("Mean severity by topic label:\n", severity_by_topic)

    return df

