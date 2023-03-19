import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS
from matplotlib.patches import Circle

# Generate sample data
X, y = make_blobs(n_samples=1000, centers=4, random_state=42)

# Perform clustering using OPTICS
clustering = OPTICS(min_samples=10, xi=.05, min_cluster_size=.05)

# Fit the data
clustering.fit(X)

# Get the cluster labels for each data point
labels = clustering.labels_

# Get the unique cluster labels
unique_labels = np.unique(labels)

# Plot the clusters with circles
fig, ax = plt.subplots(figsize=(10, 6))

for label in unique_labels:
    # Plot the data points for the current cluster
    mask = labels == label
    ax.scatter(X[mask, 0], X[mask, 1], label=label)
    
    # Add a circle around the current cluster
    center = np.mean(X[mask, :], axis=0)
    radius = np.max(np.linalg.norm(X[mask, :] - center, axis=1))
    circle = Circle(center, radius, alpha=0.2)
    ax.add_artist(circle)

# Add a legend and show the plot
ax.legend()
plt.show()
