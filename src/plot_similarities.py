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
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution of Number of Clusters
ax1 = axes[0, 0]
ax1.hist(num_clusters, bins=range(1, max(num_clusters) + 2), edgecolor='black', alpha=0.7, color='steelblue', align='left')
ax1.axvline(np.mean(num_clusters), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(num_clusters):.2f}')
ax1.set_title('Distribution of Number of Clusters', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Frequency')
ax1.set_xticks(range(1, max(num_clusters) + 1))
ax1.legend()

# 2. Distribution of Semantic Entropy
ax2 = axes[0, 1]
ax2.hist(entropies, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(np.mean(entropies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(entropies):.3f}')
ax2.axvline(np.median(entropies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(entropies):.3f}')
ax2.set_title('Distribution of Semantic Entropy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Semantic Entropy')
ax2.set_ylabel('Frequency')
ax2.legend()

# 3. Distribution of Confidence Score
ax3 = axes[1, 0]
ax3.hist(conf_scores, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
ax3.axvline(np.mean(conf_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(conf_scores):.3f}')
ax3.axvline(np.median(conf_scores), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(conf_scores):.3f}')
ax3.set_title('Distribution of Confidence Score', fontsize=12, fontweight='bold')
ax3.set_xlabel('Confidence Score')
ax3.set_ylabel('Frequency')
ax3.legend()

# 4. Scatter: Entropy vs Confidence
ax4 = axes[1, 1]
scatter = ax4.scatter(entropies, conf_scores, c=num_clusters, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
ax4.set_title('Entropy vs Confidence Score', fontsize=12, fontweight='bold')
ax4.set_xlabel('Semantic Entropy')
ax4.set_ylabel('Confidence Score')
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Number of Clusters')

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
