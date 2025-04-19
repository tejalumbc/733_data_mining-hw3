import pandas as pd
import matplotlib.pyplot as plt
from kmeans_old_faithful import KMeans  # Import custom class

# Load data
df = pd.read_csv('faithful.csv')
X = df[['eruptions', 'waiting']].values

# Run K-Means
kmeans = KMeans(n_clusters=2, max_iter=20)
kmeans.fit(X)

# Plot objective function
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(kmeans.objective_history)+1), 
         kmeans.objective_history,
         marker='o', 
         color='#1f77b4',
         linewidth=2)
plt.title('K-Means Convergence (Objective Function)', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Sum of Squared Distances', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, len(kmeans.objective_history)+1))

# Mark convergence point
if len(kmeans.objective_history) < kmeans.max_iter:
    plt.axvline(x=len(kmeans.objective_history), color='red', linestyle=':')
    plt.annotate(f'Converged at iteration {len(kmeans.objective_history)}', 
                xy=(len(kmeans.objective_history), kmeans.objective_history[-1]),
                xytext=(10, 10), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300)
plt.show()