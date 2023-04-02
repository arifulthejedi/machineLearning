import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load data
data = pd.read_csv('data.csv')

# Preprocess data
X = data.iloc[:, :-1]  # select all columns except the last one
y = data.iloc[:, -1]   # select the last column as labels
X_scaled = (X - X.mean()) / X.std()  # standardize the features

# Create T-SNE model
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)

# Fit T-SNE model to data
X_tsne = tsne.fit_transform(X_scaled)

# Create scatter plot
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title("crime.csv TSNE plot")
plt.colorbar()
plt.show()
