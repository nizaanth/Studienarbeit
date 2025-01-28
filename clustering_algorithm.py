import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset from the .txt file
file_path = "dataset/production_comments.txt"  # Adjust the file path as needed
data = pd.read_csv(file_path, delimiter=",", header=0)

# Verify the dataset structure
print("Loaded Dataset:")
print(data.head())

# Step 2: Extract the 'Kommentar' column for clustering
comments = data["Kommentar"].tolist()

# Step 3: Load the GBERT model and tokenizer using TensorFlow
print("\nLoading GBERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')  # GBERT tokenizer
model = TFBertModel.from_pretrained('bert-base-german-cased')  # GBERT model for TensorFlow

# Step 4: Generate embeddings for the comments
def generate_embeddings(comments):
    embeddings = []
    for comment in comments:
        # Tokenize the comment
        inputs = tokenizer(comment, return_tensors="tf", truncation=True, padding=True, max_length=512)
        # Get the model output (we take the embeddings from the last hidden state)
        outputs = model(**inputs)
        # Mean Pooling to get a single vector representation for the comment
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        # Cast attention_mask to float32 for compatibility with the hidden states
        attention_mask = tf.cast(attention_mask, tf.float32)
        # Apply attention mask to ignore padding tokens in the averaging
        masked_hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
        sentence_embedding = tf.reduce_sum(masked_hidden_states, axis=1) / tf.reduce_sum(attention_mask, axis=1, keepdims=True)
        embeddings.append(sentence_embedding.numpy().flatten())
    return np.array(embeddings)

print("\nGenerating embeddings...")
embeddings = generate_embeddings(comments)

# Step 5: Evaluate clustering algorithms
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=5, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=5),
    "SpectralClustering": SpectralClustering(n_clusters=5, affinity='nearest_neighbors'),
    "Birch": Birch(n_clusters=5)
}

# Initialize dictionary to store silhouette scores
silhouette_scores = {}

# Loop through clustering algorithms and calculate silhouette score
for name, algorithm in clustering_algorithms.items():
    print(f"\nApplying {name}...")
    try:
        labels = algorithm.fit_predict(embeddings)
        # Check if DBSCAN produced any noise points (labels == -1)
        if len(set(labels)) > 1:  # At least 2 clusters
            score = silhouette_score(embeddings, labels)
            silhouette_scores[name] = score
            print(f"Silhouette Score for {name}: {score:.2f}")
        else:
            print(f"{name} did not produce valid clusters.")
    except Exception as e:
        print(f"Error applying {name}: {str(e)}")

# Step 6: Plot silhouette scores for each algorithm
print("\nSilhouette Scores for different clustering algorithms:")
for name, score in silhouette_scores.items():
    print(f"{name}: {score:.2f}")

# Plot the silhouette scores
plt.figure(figsize=(8, 6))
algorithms = list(silhouette_scores.keys())
scores = list(silhouette_scores.values())
plt.barh(algorithms, scores, color='skyblue')
plt.xlabel('Silhouette Score')
plt.title('Silhouette Scores for Different Clustering Algorithms')
plt.show()

# Step 7: Visualize the embeddings using PCA for the best clustering algorithm
best_algorithm_name = max(silhouette_scores, key=silhouette_scores.get)  # Get the best algorithm
best_algorithm = clustering_algorithms[best_algorithm_name]

print(f"\nVisualizing embeddings using PCA for the best algorithm: {best_algorithm_name}...")
labels = best_algorithm.fit_predict(embeddings)

# PCA for 2D visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Create plots folder if it doesn't exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Generate PCA plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title(f"PCA Visualization of Text Clusters ({best_algorithm_name})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Save the plot
plot_path = os.path.join(plots_folder, f"pca_clusters_{best_algorithm_name}.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"PCA plot saved to {plot_path}")
