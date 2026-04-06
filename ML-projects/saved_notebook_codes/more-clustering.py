# first we try to understand with a small synthetic example
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Create Synthetic Data (3 nested clusters)
# Group A (bottom left), Group B (slightly above A), Group C (top right)
X = np.array([
    [1, 1], [1.1, 1.1], [0.9, 1.2],    # Group A
    [1.5, 1.8], [1.6, 1.7], [1.4, 1.9], # Group B (Near A)
    [5, 5], [5.1, 4.9], [4.8, 5.2], [5.2, 5.1] # Group C (Far)
])

# Perform Hierarchical Clustering
# 'ward' linkage minimizes the variance of clusters being merged
Z = linkage(X, method='ward')

# Plotting
plt.figure(figsize=(10, 5))

# Plot 1: The Scatter Plot (What the data looks like)
plt.subplot(1, 2, 1)
plt.scatter(X[:,0], X[:,1], c='blue')
plt.title("Synthetic Data Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot 2: The Dendrogram (The 'Tree')
plt.subplot(1, 2, 2)
dendrogram(Z)
plt.title("Hierarchical Dendrogram")
plt.xlabel("Data Point Index")
plt.ylabel("Distance (Dissimilarity)")

plt.tight_layout()
plt.show()

# THIS time we will put it into tree
from scipy.cluster.hierarchy import to_tree
import pandas as pd

# Convert the matrix into a tree object and a list of all nodes
rootnode, nodelist = to_tree(Z, rd=True)

# blank map
parent_map = {node.id: None for node in nodelist}

# Populate the map by traversing the nodes
for node in nodelist:
    if not node.is_leaf():
        # The current node is the parent of its left and right children
        parent_map[node.left.id] = node.id
        parent_map[node.right.id] = node.id

# Create the final flat structure for your database
hierarchy_data = []
for node in nodelist:
    hierarchy_data.append({
        "node_id": node.id,
        "parent_id": parent_map[node.id],  # Now populated
        "is_leaf": node.is_leaf(),
        "distance": node.dist if not node.is_leaf() else 0,
        "count": node.count # Number of original points in this cluster
    })

# Convert to DataFrame
df_hierarchy = pd.DataFrame(hierarchy_data)

# show it
df_hierarchy.describe()
df_hierarchy.info()
df_hierarchy.head(25)


# THIS time we will apply on some real data
Spellman_df = pd.read_csv('../data/clustering/Spellman.csv')

Spellman_df.describe()
Spellman_df.info()
Spellman_df.head(5)

# drop first text column
cleaned_Spellman_df = Spellman_df.drop(['time'], axis = 1)

cleaned_Spellman_df.describe()
cleaned_Spellman_df.info()
cleaned_Spellman_df.head(5)

# Plotting, THIS time we will go with some different techniques to handle around 5K data

from sklearn.decomposition import PCA
import seaborn as sns

# Perform Hierarchical Clustering
# 'ward' linkage minimizes the variance of clusters being merged
Z = linkage(cleaned_Spellman_df, method='ward')

# Reduce dimensions to 2
pca = PCA(n_components=2)
components = pca.fit_transform(cleaned_Spellman_df) # Use the 1/0 numeric data

# dataframe for scatter plot
# Add PC1 and PC2 to your dataframe
scatter_plot_df = pd.DataFrame({
   'PC1' : components[:, 0],
   'PC2' : components[:, 1]
   })

# df_raw['PC1'] = components[:, 0]
# df_raw['PC2'] = components[:, 1]

# Plot the clusters found by the hierarchy (using a 'cut' at k=3)
from scipy.cluster.hierarchy import fcluster
scatter_plot_df['cluster_id'] = fcluster(Z, t=3, criterion='maxclust')

plt.figure(figsize=(8,6))
sns.scatterplot(data=scatter_plot_df, x='PC1', y='PC2', hue='cluster_id', palette='viridis')
plt.title("2D Projection of Multi-Column Data")
plt.show()

# dendrogram for 5K data
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(10, 7))
dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last 'p' merged clusters
    p=20,                   # Show only the top 20 nodes
    show_leaf_counts=True,  # Show how many points are in each leaf
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,   # Draw black dots at the heights of those hidden merges
)
plt.title("Truncated Dendrogram (Top 20 Clusters)")
plt.xlabel("Cluster Size (number of points)")
plt.ylabel("Distance")
plt.show()


# FRUAD transaction details
fraud_transactions_df = pd.read_csv('../data/clustering/Fraud Detection Transactions Dataset.csv')

fraud_transactions_df.describe()
fraud_transactions_df.info()
fraud_transactions_df.head(5)

# it's labelled data, so we will first drop label column and then try to cluster
# TODO later on we can explore with supervised classification algorithm

cleaned_fraud_transactions_df = fraud_transactions_df.drop(['Transaction_ID', 'Fraud_Label'], axis = 1)

cleaned_fraud_transactions_df.describe()
cleaned_fraud_transactions_df.info()
cleaned_fraud_transactions_df.head(5)


# apply DBSCAN on data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# identify categorical and numerical columns
cat_cols = [col for col in cleaned_fraud_transactions_df.columns if cleaned_fraud_transactions_df[col].dtype == 'str']
num_cols = [col for col in cleaned_fraud_transactions_df.columns if col not in cat_cols]

# Build the Preprocessing Pipeline
# We apply Ordinal Encoding to categories and Scaling to everything
preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()) # Final scale to ensure all features have equal weight
])

# Process Data and Cluster
X_transformed = pipeline.fit_transform(cleaned_fraud_transactions_df)

# Fit DBSCAN (Note: DBSCAN is often used outside the pipeline for fit_predict)
# These hyperparameters we can optimize later on by some iterative metric evaluation
dbscan = DBSCAN(eps=1.0, min_samples=2)
labels = dbscan.fit_predict(X_transformed)

# Evaluation Metrics
# INTRODUCING safe calculations
# Filter out the noise (-1)
core_labels = labels[labels != -1]
core_samples = X_transformed[labels != -1]

# Count unique clusters (excluding noise)
unique_clusters = len(set(core_labels))

# Safe Evaluation
if unique_clusters > 1:
    sil = silhouette_score(X_transformed[labels != -1], labels[labels != -1])
    dbi = davies_bouldin_score(X_transformed[labels != -1], labels[labels != -1])
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")
else:
    print(f"Evaluation Skipped: Found {unique_clusters} cluster(s).")
    print("Action: Try decreasing 'eps' or decreasing 'min_samples' to split the data.")
    
    
# K-DISTANT plot we will use to find optimal hyperparameters eps and n_samples
from sklearn.neighbors import NearestNeighbors

# SET HYPERPARAMETER: min_samples (k)
# Rule of thumb: k = 2 * number of dimensions
k_neighbors = 2 * X_transformed.shape[1] 

# CALCULATE K-NEAREST NEIGHBORS
# We use the scaled feature matrix 'X_transformed' from earlier
neigh = NearestNeighbors(n_neighbors=k_neighbors)
nbrs = neigh.fit(X_transformed)
distances, indices = nbrs.kneighbors(X_transformed)

# SORT DISTANCES
# We take the distance to the k-th neighbor (the last column)
# and sort them from smallest to largest
k_distances = np.sort(distances[:, k_neighbors - 1], axis=0)

# PLOT THE K-DISTANCE GRAPH
plt.figure(figsize=(10, 6))
plt.plot(k_distances, color='blue', lw=2)
plt.title(f'K-Distance Plot (min_samples={k_neighbors})', fontsize=14)
plt.xlabel('Data Points (Sorted by Distance)', fontsize=12)
plt.ylabel(f'Distance to {k_neighbors}-th Neighbor', fontsize=12)

# Visually identify the 'elbow' where the slope sharply increases
# Let's say the elbow is at y=0.3 for this example
plt.axhline(y=0.3, color='red', linestyle='--', label='Optimal Eps (Elbow)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# WE HE GOT K = 2 * 19 (dimensions) and eps takes sharp vertical curve at point around eps = 4
# NOW we will check for manual eps adjustment

# Range of potential eps values to test (based on your K-distance plot 1.0 to 5.0)
eps_values = np.arange(1.0, 4.5, 0.5) 
min_samples = 2 * X_transformed.shape[1]

for e in eps_values:
    db = DBSCAN(eps=e, min_samples=min_samples).fit(X_transformed)
    labels = db.labels_
    
    # Only evaluate if we have more than 1 cluster (excluding noise)
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        sil = silhouette_score(X_transformed[labels != -1], labels[labels != -1])
        dbi = davies_bouldin_score(X_transformed[labels != -1], labels[labels != -1])
        print(f"Eps: {e:.1f} | Silhouette Score: {sil:.3f} | Davies-Bouldin Index: {dbi:.3f}")
    else:
        print(f"Eps: {e:.1f} | Not enough clusters to evaluate.")
        

# THIS time we will focus on PCA before dunning DBscan
from sklearn.decomposition import PCA

# Scale first (PCA and DBSCAN require this!)
# already StandardScaler data will have no impact again, harmless
X_scaled = StandardScaler().fit_transform(X_transformed)

# 2. Reduce 19 dimensions to 5 (keeping most of the 'meaning')
pca = PCA(n_components=5) 
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained by 5 components: {sum(pca.explained_variance_ratio_):.2%}")

# New 'min_samples' based on 5 dimensions
new_min_samples = 2 * X_pca.shape[1]

# Run DBSCAN on the reduced data
db = DBSCAN(eps=1.5, min_samples=new_min_samples).fit(X_pca)

labels = dbscan.fit_predict(X_pca)

# Evaluation Metrics
# INTRODUCING safe calculations
# Filter out the noise (-1)
core_labels = labels[labels != -1]
core_samples = X_pca[labels != -1]

# Count unique clusters (excluding noise)
unique_clusters = len(set(core_labels))

# Safe Evaluation
if unique_clusters > 1:
    sil = silhouette_score(X_pca[labels != -1], labels[labels != -1])
    dbi = davies_bouldin_score(X_pca[labels != -1], labels[labels != -1])
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")
else:
    print(f"Evaluation Skipped: Found {unique_clusters} cluster(s).")
    print("Action: Try decreasing 'eps' or decreasing 'min_samples' to split the data.")
    
    
# K-DISTANT plot we will use to find optimal hyperparameters eps and n_samples

# SET HYPERPARAMETER: min_samples (k)
# Rule of thumb: k = 2 * number of dimensions
k_neighbors = 2 * X_pca.shape[1] 

# CALCULATE K-NEAREST NEIGHBORS
# We use the scaled feature matrix 'X_pca' from earlier
neigh = NearestNeighbors(n_neighbors=k_neighbors)
nbrs = neigh.fit(X_pca)
distances, indices = nbrs.kneighbors(X_pca)

# SORT DISTANCES
# We take the distance to the k-th neighbor (the last column)
# and sort them from smallest to largest
k_distances = np.sort(distances[:, k_neighbors - 1], axis=0)

# PLOT THE K-DISTANCE GRAPH
plt.figure(figsize=(10, 6))
plt.plot(k_distances, color='blue', lw=2)
plt.title(f'K-Distance Plot (min_samples={k_neighbors})', fontsize=14)
plt.xlabel('Data Points (Sorted by Distance)', fontsize=12)
plt.ylabel(f'Distance to {k_neighbors}-th Neighbor', fontsize=12)

# Visually identify the 'elbow' where the slope sharply increases
# Let's say the elbow is at y=0.3 for this example
plt.axhline(y=0.3, color='red', linestyle='--', label='Optimal Eps (Elbow)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# NOW we will run HDBscan on PCA data
from sklearn.cluster import HDBSCAN

# RUN HDBSCAN
# min_cluster_size: The minimum size you want your groups to be.
# min_samples: Controls how 'conservative' the clustering is (defaults to min_cluster_size).
clusterer = HDBSCAN(min_cluster_size=50, min_samples=new_min_samples, gen_min_span_tree=True)
labels = clusterer.fit_predict(X_pca)

# EVALUATION (Excluding Noise -1)
mask = labels != -1
if len(np.unique(labels[mask])) > 1:
    sil = silhouette_score(X_pca[mask], labels[mask])
    dbi = davies_bouldin_score(X_pca[mask], labels[mask])
    
    print(f"Clusters found: {len(np.unique(labels[mask]))}")
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index: {dbi:.3f}")
    print(f"Fraud/Noise detected: {np.sum(labels == -1)} transactions")


# UNDERSTANDING how finding K in K means by applying hierarchy clustering and cut using dandogram

#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

# Generate synthetic data with 3 clear clusters
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.60, random_state=42)

# Perform Hierarchical Clustering
# 'ward' linkage minimizes the variance of clusters being merged
Z = linkage(X, method='ward')

# Define the threshold distance to "cut" the tree
# Visually, we look for the largest vertical gap to place this line
cut_distance = 7.0

# Visualization
plt.figure(figsize=(10, 6))
plt.title("Dendrogram with Optimal K Cut-off")
plt.xlabel("Sample Index")
plt.ylabel("Distance (Dissimilarity)")

# Plotting the dendrogram
# color_threshold automatically colors clusters separated by the cut
dendrogram(Z, color_threshold=cut_distance, above_threshold_color='grey')

# Drawing the horizontal cut line
plt.axhline(y=cut_distance, color='r', linestyle='--', label=f'Cut Line (d={cut_distance})')
plt.legend()
plt.show()

# Extracting the number of clusters (K)
labels = fcluster(Z, t=cut_distance, criterion='distance')
print(f"Total Clusters (K) identified: {len(np.unique(labels))}")