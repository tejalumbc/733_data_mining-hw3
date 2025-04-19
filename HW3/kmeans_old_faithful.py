import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('faithful.csv')
X = df[['eruptions', 'waiting']].values

# K-Means c
class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_idx]
    
    def compute_distances(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        return centroids
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            distances = self.compute_distances(X, self.centroids)
            self.labels = self.assign_clusters(distances)
            self.centroids = self.update_centroids(X, self.labels)
            if np.linalg.norm(self.centroids - old_centroids) < self.tol:
                break
        return self

# Run K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200)
plt.title('Old Faithful Eruptions - K-Means Clustering')
plt.xlabel('Eruption Duration (minutes)')
plt.ylabel('Waiting Time (minutes)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("Cluster Centers:")
for i, center in enumerate(kmeans.centroids):
    print(f"Cluster {i+1}: Eruption = {center[0]:.2f} min, Waiting = {center[1]:.2f} min")