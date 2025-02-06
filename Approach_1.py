import pandas as pd
import numpy as np
import re
import chardet
from collections import Counter
from umap import UMAP
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from stop_words import get_stop_words

# Configuration
SEED = 42
N_CLUSTERS = 6

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding']

# Enhanced German preprocessing with manufacturing context
def preprocess_comments(texts):
    # Domain-specific stopwords
    stopwords = set([
        'Maschine', 'Produktion', 'Schicht', 'Linie', 
        'System', 'Parameter', 'Prozess', 'Teil', 'Wert'
    ] + get_stop_words('german'))
    
    processed = []
    for text in texts:
        # Clean text
        text = re.sub(r'[^a-zA-ZäöüßÄÖÜ\- ]', '', str(text))
        text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
        
        # Tokenize and filter - remove .lower() call
        tokens = [
            token for token in text.split() 
            if token.lower() not in stopwords and len(token) > 2
        ]
        
        processed.append(' '.join(tokens))
    return processed

# Load data
file_path = 'Dataset/production_comments.txt'
encoding = detect_file_encoding(file_path)
df = pd.read_csv(file_path, sep=',', encoding=encoding)

# Preprocess
processed_comments = preprocess_comments(df['Kommentar'])

# Generate embeddings
gbert = SentenceTransformer('bert-base-german-cased')
embeddings = gbert.encode(processed_comments)

# Dimensionality reduction
umap_model = UMAP(n_components=3, random_state=SEED)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Clustering
cluster_model = KMeans(n_clusters=N_CLUSTERS, random_state=SEED)
clusters = cluster_model.fit_predict(reduced_embeddings)

# Topic modeling
topic_model = BERTopic(
    language='german',
    top_n_words=8,
    nr_topics='auto',
    verbose=True
)
topics, _ = topic_model.fit_transform(processed_comments, embeddings)

# Integrate results
df['Cluster'] = clusters
df['Topic'] = topics
df['Keywords'] = df['Topic'].map(topic_model.get_topic_info().set_index('Topic')['Name'])

# Evaluation metrics
silhouette = silhouette_score(reduced_embeddings, clusters)
print(f"Silhouette Score: {silhouette:.2f}")

# Visualization
plt.figure(figsize=(12,8))
plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], 
            c=clusters, cmap='viridis', s=50)
plt.title("UMAP Projection of Manufacturing Comments")
plt.savefig('cluster_visualization.png')

# Generate topic report
topic_report = topic_model.get_topic_info()
topic_report.to_csv('topic_report.csv', index=False)

# Save results
df.to_csv('manufacturing_insights.csv', index=False)