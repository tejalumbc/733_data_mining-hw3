import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from kmeans_old_faithful import KMeans

# Load data
df = pd.read_csv('faithful.csv')
X = df[['eruptions', 'waiting']].values

# Run K-Means 
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# 
plt.figure(figsize=(10, 6), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color map
colors = cm.get_cmap('tab10', 2)

# Plot clustered data
scatter = plt.scatter(
    X[:, 0], X[:, 1], 
    c=kmeans.labels, 
    cmap=colors,
    s=50,
    alpha=0.8,
    edgecolor='white',
    linewidth=0.5
)

# Plot centroids
centroid_plot = plt.scatter(
    kmeans.centroids[:, 0], 
    kmeans.centroids[:, 1],
    marker='X',
    s=200,
    c='red',
    edgecolor='black',
    linewidth=1,
    label='Cluster Centers'
)

# Annotate centroids
for i, center in enumerate(kmeans.centroids):
    plt.annotate(
        f'Center {i+1}\n({center[0]:.2f}, {center[1]:.2f})',
        xy=center,
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
        arrowprops=dict(arrowstyle='->')
    )

# Labels and title
plt.title('Old Faithful Eruptions - K-Means Clustering (k=2)\n', fontsize=14, pad=20)
plt.xlabel('Eruption Duration (minutes)', fontsize=12, labelpad=10)
plt.ylabel('Waiting Time (minutes)', fontsize=12, labelpad=10)

# Legend and grid
plt.legend(handles=[*scatter.legend_elements()[0], centroid_plot], 
           labels=['Cluster 1', 'Cluster 2', 'Centers'],
           frameon=True,
           framealpha=1)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save high-quality image
plt.savefig('old_faithful_clusters.png', dpi=300, bbox_inches='tight')
plt.show()