"""
Divisive Clustering (Top-down) with similarity threshold.
"""
from sentence_transformers import util


def divisive_clustering(embeddings, similarity_threshold=0.95, min_clusters=3):
    """
    Divisive Clustering (Top-down): Bắt đầu từ 1 cụm, chia nhỏ dần.

    Thuật toán:
    1. Bắt đầu: tất cả điểm trong 1 cụm
    2. Lặp: tìm cụm có 2 điểm xa nhất, nếu sim < threshold thì split
    3. Dừng khi: tất cả cụm đã "tight" (sim >= threshold) VÀ đã đạt min_clusters

    Args:
        embeddings: torch.Tensor shape (n_samples, embedding_dim)
        similarity_threshold: Ngưỡng similarity để giữ nguyên cụm (không split nữa)
        min_clusters: Số cụm tối thiểu phải đạt được

    Returns:
        clusters: List of lists, mỗi list chứa indices của các phần tử trong cụm
    """
    n = embeddings.shape[0]

    if n == 1:
        return [[0]]

    # Bước 1: Tất cả điểm trong 1 cụm
    clusters = [list(range(n))]

    # Tính similarity matrix một lần
    sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    while True:
        # Tìm cụm cần split (có 2 điểm xa nhất với sim < threshold)
        best_split_idx = -1
        best_min_sim = 1.0
        best_i1, best_i2 = -1, -1

        for c_idx, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue

            # Tìm 2 điểm xa nhất trong cụm
            min_sim = 1.0
            min_i, min_j = 0, 1
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    sim = sim_matrix[cluster[i]][cluster[j]].item()
                    if sim < min_sim:
                        min_sim = sim
                        min_i, min_j = i, j

            # Nếu cụm này có sim nhỏ nhất và cần split
            if min_sim < best_min_sim:
                best_min_sim = min_sim
                best_split_idx = c_idx
                best_i1, best_i2 = min_i, min_j

        # Điều kiện dừng:
        # - Không có cụm nào cần split (tất cả tight) VÀ đã đạt min_clusters
        # - Hoặc tất cả cụm chỉ có 1 phần tử
        if best_split_idx == -1:
            break

        # Nếu đã đạt min_clusters và tất cả cụm đã tight
        if len(clusters) >= min_clusters and best_min_sim >= similarity_threshold:
            break

        # Split cụm được chọn
        cluster = clusters[best_split_idx]
        centroid1 = embeddings[cluster[best_i1]]
        centroid2 = embeddings[cluster[best_i2]]

        new_cluster1, new_cluster2 = [], []
        for idx in cluster:
            sim1 = sim_matrix[cluster[best_i1]][idx].item()
            sim2 = sim_matrix[cluster[best_i2]][idx].item()
            if sim1 >= sim2:
                new_cluster1.append(idx)
            else:
                new_cluster2.append(idx)

        # Thay cụm cũ bằng 2 cụm mới
        clusters.pop(best_split_idx)
        if new_cluster1:
            clusters.append(new_cluster1)
        if new_cluster2:
            clusters.append(new_cluster2)

    return clusters
