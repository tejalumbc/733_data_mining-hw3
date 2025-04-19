import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('faithful.csv')

#  scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['waiting'], df['eruptions'], 
            c='steelblue', alpha=0.7, edgecolors='w', linewidth=0.5)

#  labels and title
plt.xlabel('Waiting Time to Next Eruption (minutes)', fontsize=12)
plt.ylabel('Eruption Duration (minutes)', fontsize=12)
plt.title('Old Faithful Geyser Eruptions\nDuration vs. Waiting Time', 
          fontsize=14, pad=20)

# Add grid  layout
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

#  plot
plt.show()