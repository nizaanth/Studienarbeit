import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM, BertConfig
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import wandb  # Import WandB
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger

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

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))

# Define callbacks for early stopping and model checkpointing
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

# Step 9: Perform K-Means clustering
num_clusters = 7
kmeans = KMeans(n_clusters=num_clusters, random_state=SEED)
labels = kmeans.fit_predict(embeddings)

# Step 10: Add cluster labels to the original dataset
data["Cluster"] = labels

# Step 11: Save the clustered data
output_path = "dataset/clustered_comments_fine_tuned_gbert.csv"
data.to_csv(output_path, index=False)
print(f"\nClustered data saved to {output_path}")

# Step 12: Calculate silhouette score
silhouette_avg = silhouette_score(embeddings, labels)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Step 13: Visualize clusters using PCA
print("\nVisualizing clusters with PCA...")
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Create plots folder if it doesn't exist
plots_folder = "plots"
os.makedirs(plots_folder, exist_ok=True)

# Generate PCA plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("PCA Visualization of Text Clusters (Fine-tuned GBERT)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig(os.path.join(plots_folder, "pca_clusters_fine_tuned_gbert.png"), dpi=300, bbox_inches="tight")
plt.close()

print("PCA plot saved.")