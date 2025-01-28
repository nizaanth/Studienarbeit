import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset from the .txt file
file_path = "Dataset/production_comments.txt"  # Adjust the file path as needed
data = pd.read_csv(file_path, delimiter=",", header=0)

# Verify the dataset structure
print("Loaded Dataset:")
print(data.head())

# Step 2: Extract the 'Kommentar' column for clustering
comments = data["Kommentar"].tolist()

# Step 3: Load the MiniLM-L12-v2 model
print("\nLoading MiniLM-L12-v2 model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


# Step 4: Generate embeddings for the comments
print("\nGenerating embeddings...")
embeddings = model.encode(comments, show_progress_bar=True)

# Step 5: Perform K-Means clustering
num_clusters = 5  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Step 6: Add cluster labels to the original dataset
data["Cluster"] = labels

# Step 7: Analyze and save the clustered data
print("\nClustered Data Sample:")
print(data.head())

output_path = "dataset/clustered_comments.csv"
data.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

# Step 8: Calculate silhouette score to evaluate clustering
silhouette_avg = silhouette_score(embeddings, labels)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Step 9: Visualize clusters using t-SNE and save the plot
print("\nVisualizing clusters...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(embeddings)

# Create plots folder if it doesn't exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Generate t-SNE plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("t-SNE Visualization of Text Clusters")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

# Save the plot
plot_path = os.path.join(plots_folder, "tsne_clusters.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"t-SNE plot saved to {plot_path}")
