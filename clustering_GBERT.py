import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset from the .txt file
file_path = "dataset/production_comments.txt"  # Adjust the file path as needed
data = pd.read_csv(file_path, delimiter=",", header=0)

# Verify the dataset structure
print("Loaded Dataset:")
print(data.head())

# Check for missing values
if data["Kommentar"].isnull().any():
    print("Warning: There are missing values in the 'Kommentar' column.")
    data = data.dropna(subset=["Kommentar"])  # Optionally drop them

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

# Step 5: Perform K-Means clustering using scikit-learn
num_clusters = 5  # Set the desired number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Step 6: Add cluster labels to the original dataset
data["Cluster"] = labels

# Step 7: Save the clustered data
output_path = "dataset/clustered_comments_gbert_tensorflow.csv"
data.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

# Step 8: Calculate silhouette score to evaluate clustering
silhouette_avg = silhouette_score(embeddings, labels)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Step 9: Visualize clusters using PCA and save the plot
print("\nVisualizing clusters with PCA...")
pca = PCA(n_components=2)  # Reduce to 2D for visualization
reduced_embeddings = pca.fit_transform(embeddings)

# Create plots folder if it doesn't exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Generate PCA plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("PCA Visualization of Text Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

# Save the plot
plot_path = os.path.join(plots_folder, "pca_clusters_gbert_tensorflow.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"PCA plot saved to {plot_path}")
