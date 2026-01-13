import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
data_path = Path(__file__).parent / "clusters_only.jsonl"
data = []

with open(data_path, 'r') as f:
    for line in f:
        obj = json.loads(line)
        data.append(obj)

# Extract metrics
num_clusters = [len(d['clusters']) for d in data]
entropies = [d['semantic_entropy'] for d in data]
conf_scores = [d['conf_score'] for d in data]
largest_cluster_sizes = [max(len(c) for c in d['clusters']) for d in data]

print(f"Total samples: {len(data)}")

# Set style
sns.set_style("white")

# Create single plot
fig, ax = plt.subplots(figsize=(10, 6))

# Distribution of Number of Clusters
ax.hist(num_clusters, bins=range(1, max(num_clusters) + 2), edgecolor='black', alpha=0.7, color='steelblue', align='left')
ax.axvline(np.mean(num_clusters), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(num_clusters):.2f}')
ax.set_title('Distribution of Number of Clusters', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Clusters', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_xticks(range(1, max(num_clusters) + 1))
ax.legend()
ax.grid(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "clusters_distribution.png", dpi=300, bbox_inches='tight')
print("✓ Saved: clusters_distribution.png")
plt.show()

# Print statistics
print("\n" + "="*50)
print("CLUSTERING STATISTICS")
print("="*50)
print(f"Total samples: {len(data)}")
print(f"\nNumber of Clusters:")
print(f"  Mean: {np.mean(num_clusters):.2f}")
print(f"  Median: {np.median(num_clusters):.2f}")
print(f"  Min: {np.min(num_clusters)}")
print(f"  Max: {np.max(num_clusters)}")
print(f"\nSemantic Entropy:")
print(f"  Mean: {np.mean(entropies):.4f}")
print(f"  Median: {np.median(entropies):.4f}")
print(f"  Std: {np.std(entropies):.4f}")
print(f"\nConfidence Score:")
print(f"  Mean: {np.mean(conf_scores):.4f}")
print(f"  Median: {np.median(conf_scores):.4f}")
print(f"  Std: {np.std(conf_scores):.4f}")
print(f"\nConfidence >= 0.5: {sum(1 for c in conf_scores if c >= 0.5)} ({sum(1 for c in conf_scores if c >= 0.5)/len(conf_scores)*100:.1f}%)")
print(f"Confidence >= 0.7: {sum(1 for c in conf_scores if c >= 0.7)} ({sum(1 for c in conf_scores if c >= 0.7)/len(conf_scores)*100:.1f}%)")
print(f"Single cluster (conf=1.0): {sum(1 for n in num_clusters if n == 1)} ({sum(1 for n in num_clusters if n == 1)/len(num_clusters)*100:.1f}%)")
