# Standard library imports
import os
from typing import List, Dict, Optional

# Third-party imports: Data processing and analysis
import numpy as np
import pandas as pd

# Third-party imports: Machine Learning & Deep Learning
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score, 
    calinski_harabasz_score
)

# Third-party imports: Transformers
from transformers import (
    BertTokenizer, 
    TFBertForMaskedLM, 
    BertConfig
)

# Third-party imports: Visualization
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Third-party imports: Deep Learning tools
import wandb
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from wandb.integration.keras import WandbMetricsLogger

# Optional visualization imports (uncomment if using UMAP/t-SNE)
# import umap
# from sklearn.manifold import TSNE

SEED = 42
EPOCHS = 100
WANDB_KEY = '45d0f1022dabf560fcb7388b00ee3b6a378d54a8'
wandb.login(key=WANDB_KEY)



# Load the dataset from the .txt file
file_path = "dataset/production_comments.txt"  # Adjust the file path as needed
data = pd.read_csv(file_path, delimiter=",", header=None, names=["Kommentar"])

# Check for missing values
if data["Kommentar"].isnull().any():
    print("Warning: There are missing values in the 'Kommentar' column. Dropping them.")
    data = data.dropna(subset=["Kommentar"])

# Prepare the comments for training
comments = data["Kommentar"].tolist()

# Load tokenizer and model with configuration for masked language modeling
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
config = BertConfig.from_pretrained('bert-base-german-cased', output_hidden_states=True)
model = TFBertForMaskedLM.from_pretrained('bert-base-german-cased', config=config)

# Prepare the dataset
def tokenize_function(comments):
    return tokenizer(comments, padding='max_length', truncation=True, max_length=512)

tokenized_comments = tokenize_function(comments)

# Create masked labels
inputs = tokenized_comments['input_ids']
labels = np.array(inputs)
mask_prob = 0.15
for i in range(len(labels)):
    input_length = np.count_nonzero(labels[i])
    num_masks = max(1, int(input_length * mask_prob))
    mask_indices = np.random.choice(range(input_length), size=num_masks, replace=False)
    labels[i][mask_indices] = tokenizer.mask_token_id

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(8)

# Add gradient clipping to prevent exploding gradients
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
# Define callbacks for early stopping and model checkpointing and learing rate scheduling
# Callbacks

cb_autosave = ModelCheckpoint("best_GBERT_FT.h5",
                              mode="max",
                              save_best_only=True,
                              monitor="loss",
                              verbose=1)

cb_early_stop = EarlyStopping(patience=10,
                              verbose=1,
                              mode="auto",
                              monitor="loss")

lr_scheduler =  ReduceLROnPlateau(monitor='loss',
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-6)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="GBERT Finetuning",
    name="GPU_1",

    # track hyperparameters and run metadata
    config={
    "architecture": "Transformers",
    "dataset": "production_comments-txt"
    }
)

cb_wandb = WandbMetricsLogger()

callbacks = [cb_autosave, cb_early_stop, cb_wandb]

# Fine-tune the model with additional callbacks
model.fit(dataset, epochs=EPOCHS, callbacks=callbacks)

# Finish the WandB run
wandb.finish()

# Generate embeddings and perform clustering as described in your code...
# Step 8: Generate embeddings from the fine-tuned model
def generate_embeddings(comments):
    embeddings = []
    for comment in comments:
        input_ids = tokenizer(comment, return_tensors="tf", truncation=True, padding=True, max_length=512)
        
        # Pass inputs through the model
        outputs = model(input_ids)

        # Extract the last hidden state (index -1 for the last layer)
        last_hidden_state = outputs.hidden_states[-1]

        # Calculate mean of hidden states for the sentence embedding
        sentence_embedding = tf.reduce_mean(last_hidden_state, axis=1).numpy().flatten()

        embeddings.append(sentence_embedding)
    return np.array(embeddings)

print("Generating embeddings using the fine-tuned model...")
embeddings = generate_embeddings(comments)

# Step 9: Determine optimal number of clusters using elbow method
def find_optimal_clusters(embeddings, max_clusters=15):
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
        
    # Find elbow point
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    optimal_k = kn.elbow
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('plots/elbow_curve.png')
    plt.close()
    
    return optimal_k

optimal_k = find_optimal_clusters(embeddings)
print(f"Optimal number of clusters: {optimal_k}")


# Step 10: Add cluster labels to the original dataset

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(embeddings)
# Now assign these labels to your dataframe
data["Cluster"] = labels


# Step 11: Save the clustered data
output_path = "dataset/clustered_comments_fine_tuned_gbert.csv"
data.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

# Step 12: Calculate metrics
print("\nCalculating metrics...")
silhouette_avg = silhouette_score(embeddings, labels)
db_score = davies_bouldin_score(embeddings, labels)
ch_score = calinski_harabasz_score(embeddings, labels)

print(f"\nSilhouette Score: {silhouette_avg:.2f}")
print(f"Davies-Bouldin Score: {db_score:.2f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")


# Step 13: Visualize clusters 
# Add UMAP visualization (better for high-dimensional data than PCA)
import umap

def visualize_clusters(embeddings, labels, method='both'):
    if method in ['umap', 'both']:
        # UMAP visualization
        reducer = umap.UMAP(random_state=SEED)
        umap_embeddings = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                            c=labels, cmap="viridis", s=50)
        plt.colorbar(scatter, label="Cluster")
        plt.title("UMAP Visualization of Text Clusters")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.savefig("plots/umap_clusters.png", dpi=300, bbox_inches="tight")
        plt.close()

    if method in ['tsne', 'both']:
        # t-SNE visualization
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=SEED)
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                            c=labels, cmap="viridis", s=50)
        plt.colorbar(scatter, label="Cluster")
        plt.title("t-SNE Visualization of Text Clusters")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.savefig("plots/tsne_clusters.png", dpi=300, bbox_inches="tight")
        plt.close()

visualize_clusters(embeddings, labels, method='both')


# Create plots folder if it doesn't exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Add cluster analysis
def analyze_clusters(data, labels):
    for i in range(max(labels) + 1):
        cluster_texts = data[data['Cluster'] == i]['Kommentar'].tolist()
        print(f"\nCluster {i} ({len(cluster_texts)} comments):")
        # Print a few example comments from each cluster
        print("Sample comments:")
        for text in cluster_texts[:3]:
            print(f"- {text[:100]}...")
        
        # Calculate average comment length per cluster
        avg_length = sum(len(text.split()) for text in cluster_texts) / len(cluster_texts)
        print(f"Average comment length: {avg_length:.1f} words")
        

analyze_clusters(data, labels)
