from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd


df = pd.read_csv('preferences.csv')
# df = pd.read_csv('data.csv')

# Prepare feature matrix and article names
categories = df[['category_1', 'category_2', 'category_3']].values
articles = df['article'].tolist()

x = categories[:, 0]
y = categories[:, 1]
z = categories[:, 2]

# --- Step 1: Cluster using DBSCAN ---
dbscan = DBSCAN(eps=2, min_samples=2)
cluster_labels = dbscan.fit_predict(categories)

# Assign colors
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
colors = plt.colormaps.get_cmap('tab10')

# --- Step 2: Query Point ---
query_point = np.array([1, 1, 7])

# Find nearest neighbors manually
distances = distance.cdist([query_point], categories, 'cosine')[0]

# --- Step 3: Sort and filter unique articles ---
sorted_indices = np.argsort(distances)
seen_articles = set()
unique_indices = []

for idx in sorted_indices:
    article_name = articles[idx]
    if article_name not in seen_articles:
        seen_articles.add(article_name)
        unique_indices.append(idx)
    if len(unique_indices) >= 3:  # Choose top-K unique
        break

# Now nearest_indices are guaranteed unique
nearest_indices = np.array(unique_indices)
nearest_labels = cluster_labels[nearest_indices]
nearest_distances = distances[nearest_indices]

# Weighted Voting: closer points have more influence (1/distance)
weights = 1 / (nearest_distances + 1e-6)

# Accumulate weights for each cluster label
weighted_votes = {}
for lbl, w in zip(nearest_labels, weights):
    if lbl in weighted_votes:
        weighted_votes[lbl] += w
    else:
        weighted_votes[lbl] = w

# Final decision
assigned_label = max(weighted_votes, key=weighted_votes.get)

# Assign color for query
query_color = 'red' if assigned_label == -1 else colors(assigned_label)

# --- Step 4: Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Define noise color (RGBA format)
noise_color = (0.5, 0.5, 0.5, 1.0)

# Plot all points
scatter = ax.scatter(x, y, z,
                     c=[colors(lbl % 10) if lbl != -1 else noise_color for lbl in cluster_labels],
                     marker='o', s=80)

# Highlight Top-K Nearest Points
ax.scatter(x[nearest_indices], y[nearest_indices], z[nearest_indices],
           color='yellow', edgecolors='black', marker='o', s=150, label='Top-K Nearest')

# Plot query point
ax.scatter(query_point[0], query_point[1], query_point[2],
           color=query_color, marker='X', s=200, label='Query Point')

# Labels and Title
ax.set_xlabel('Category 1')
ax.set_ylabel('Category 2')
ax.set_zlabel('Category 3')
ax.set_title('3D Visualization (DBSCAN + Top-K Unique Articles + Weighted Voting)')

ax.legend()

# --- Step 5: Interactive hover ---
cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    sel.annotation.set_text(articles[sel.index])

# Show
plt.show()

# --- Step 6: Print Top-K Unique Articles ---
print(f"\nTop-{len(nearest_indices)} unique nearest articles to the query point {query_point.tolist()}:")

for idx in nearest_indices:
    label_info = f"(Cluster {cluster_labels[idx]})" if cluster_labels[idx] != -1 else "(Noise)"
    print(f"- Article: {articles[idx]}, Distance: {distances[idx]:.4f} {label_info}")

# Print final weighted decision
if assigned_label == -1:
    print("\nQuery point is considered as Noise (weighted voting).")
else:
    print(f"\nQuery point assigned to Cluster {assigned_label} (based on weighted Top-{len(nearest_indices)} voting)")
