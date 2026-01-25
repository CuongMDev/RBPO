"""
RBPO (Robust BPO) Pipeline - Chỉ tối ưu prompt:
1. Nhận prompt gốc
2. Sinh M prompt tối ưu bằng BPO
3. Phân cụm prompts bằng AgglomerativeClustering (dựa trên semantic similarity)
4. Chọn đại diện cho mỗi cụm (median similarity với original)
5. Tính cross-cluster consensus score và chọn prompt tốt nhất
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from config import MODEL_CACHE_PATH, prompt_template_optimize
from utils import generate_batch

def load_models(device="cuda:0"):
    """Load tất cả models cần thiết"""
    print("Loading BPO model...")
    bpo_model = AutoModelForCausalLM.from_pretrained(
        bpo_model_path,
        cache_dir=MODEL_CACHE_PATH,
        torch_dtype=torch.float16
    ).eval().to(device)
    bpo_tokenizer = AutoTokenizer.from_pretrained(
        bpo_model_path,
        cache_dir=MODEL_CACHE_PATH,
        use_fast=False,
        legacy=True
    )
    bpo_model.config.return_dict = True

    print("Loading SBERT model...")
    sbert = SentenceTransformer(sbert_model_path, device=device, cache_folder=MODEL_CACHE_PATH)

    return bpo_model, bpo_tokenizer, sbert


def generate_optimized_prompts(prompt, bpo_model, bpo_tokenizer, m=10, device="cuda:0"):
    """Sinh M prompt tối ưu từ prompt gốc bằng BPO"""
    batch_prompts = [prompt_template_optimize.format(prompt) for _ in range(m)]
    optimized_prompts = generate_batch(
        bpo_model,
        bpo_tokenizer,
        batch_prompts,
        temperature=0.9,
        top_p=0.9,
        apply_chat_template=False,
        device=device
    )
    return optimized_prompts


def infer_bpo(prompt, bpo_model, bpo_tokenizer, device="cuda:0"):
    """
    BPO đơn giản: chỉ sinh 1 prompt tối ưu, không xử lý gì thêm.

    Args:
        prompt: prompt gốc
        bpo_model, bpo_tokenizer: model BPO để sinh prompt tối ưu
        device: GPU device

    Returns:
        dict với prompt gốc và optimized_prompt
    """
    optimized_prompts = generate_optimized_prompts(
        prompt, bpo_model, bpo_tokenizer, m=1, device=device
    )
    optimized_prompt = optimized_prompts[0]

    return {
        "prompt": prompt,
        "optimized_prompt": optimized_prompt
    }


def cluster_and_select_prompt(prompts, sbert, original_prompt, embeddings=None, distance_threshold=0.12, imp_enc=0.2):
    """
    Phân cụm prompts và chọn prompt tốt nhất (giống semantic.py).

    Args:
        prompts: list các prompts
        sbert: SentenceTransformer model
        original_prompt: prompt gốc để so sánh
        embeddings: embeddings đã tính sẵn (optional)
        distance_threshold: ngưỡng cosine distance cho clustering
        imp_enc: hệ số khuyến khích cải tiến (trừ điểm nếu giống original)

    Returns:
        dict với selected_idx, confidence, clusters, etc.
    """
    # Encode prompts nếu chưa có
    if embeddings is None:
        embeddings = sbert.encode(prompts, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()

    # Encode original prompt
    original_embedding = sbert.encode([original_prompt], convert_to_tensor=True)[0]

    # Xử lý trường hợp chỉ có 1 sample
    if len(prompts) == 1:
        return {
            "selected_idx": 0,
            "num_clusters": 1,
            "clusters": [[0]],
            "cluster_probs": [1.0],
            "largest_cluster_size": 1,
            "consensus_scores": [0.0]
        }

    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings_np)

    # Chuyển labels thành clusters
    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)
    clusters = list(clusters_dict.values())

    # Tính cluster probabilities
    n_samples = len(prompts)
    cluster_probs = [len(c) / n_samples for c in clusters]
    K = len(clusters)

    # Tìm cụm lớn nhất
    largest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
    largest_cluster = clusters[largest_cluster_idx]

    # Tính similarity với original cho từng phần tử trong cụm lớn nhất
    cluster_embeds = torch.stack([embeddings[i] for i in largest_cluster])
    sims_to_original = util.pytorch_cos_sim(original_embedding, cluster_embeds)[0]

    # Chọn phần tử có similarity MEDIAN với original
    sorted_indices = torch.argsort(sims_to_original)
    median_local_idx = sorted_indices[len(sorted_indices) // 2].item()
    best_idx = largest_cluster[median_local_idx]

    # Nếu chỉ có 1 cụm → không tính consensus
    if len(clusters) == 1:
        cluster_representatives = [best_idx]
        consensus_scores = [0.0]
    else:
        # Lưu representatives cho tất cả clusters (chọn median)
        cluster_representatives = []
        for cluster in clusters:
            if len(cluster) == 1:
                cluster_representatives.append(cluster[0])
            else:
                c_embeds = torch.stack([embeddings[i] for i in cluster])
                c_sims = util.pytorch_cos_sim(original_embedding, c_embeds)[0]
                c_sorted = torch.argsort(c_sims)
                c_median_idx = c_sorted[len(c_sorted) // 2].item()
                cluster_representatives.append(cluster[c_median_idx])

        # Cross-cluster consensus score
        consensus_scores = []
        for i, rep_idx in enumerate(cluster_representatives):
            score = 0.0
            rep_embed = embeddings[rep_idx]

            for j, other_rep_idx in enumerate(cluster_representatives):
                if i != j:
                    other_embed = embeddings[other_rep_idx]
                    sim = util.pytorch_cos_sim(rep_embed, other_embed).item()
                    score += sim

            # Trừ điểm nếu giống original (khuyến khích cải tiến)
            score -= util.pytorch_cos_sim(rep_embed, original_embedding).item() * imp_enc
            consensus_scores.append(score)

        # Chọn đại diện có consensus score cao nhất
        best_consensus_idx = max(range(len(consensus_scores)), key=lambda i: consensus_scores[i])
        best_idx = cluster_representatives[best_consensus_idx]

    return {
        "selected_idx": best_idx,
        "num_clusters": K,
        "clusters": clusters,
        "cluster_probs": cluster_probs,
        "largest_cluster_size": len(largest_cluster),
        "consensus_scores": consensus_scores
    }


def infer_rbpo(prompt, bpo_model, bpo_tokenizer, sbert, device="cuda:0"):
    """
    RBPO: Sinh M prompts tối ưu, phân cụm, chọn prompt tốt nhất.

    Args:
        prompt: prompt gốc
        bpo_model, bpo_tokenizer: model BPO
        sbert: SentenceTransformer model
        device: GPU device

    Returns:
        dict với prompt gốc, optimized_prompt được chọn, và các thông số clustering
    """
    # 1. Sinh M prompts tối ưu
    optimized_prompts = generate_optimized_prompts(
        prompt, bpo_model, bpo_tokenizer, m=M, device=device
    )

    # 2. Lọc các prompts khác với prompt gốc
    idx = [i for i, p in enumerate(optimized_prompts) if p.strip() != prompt.strip()]
    if not idx:
        idx = [0]  # Nếu tất cả giống prompt gốc, lấy cái đầu tiên

    unique_prompts = [optimized_prompts[i] for i in idx]

    # 3. Phân cụm và chọn prompt tốt nhất
    result = cluster_and_select_prompt(
        unique_prompts,
        sbert,
        original_prompt=prompt,
        distance_threshold=distance_threshold,
        imp_enc=imc_enc
    )

    # 4. Lấy prompt được chọn
    selected_prompt = unique_prompts[result["selected_idx"]]

    return {
        "prompt": prompt,
        "optimized_prompt": selected_prompt,
        "bpo_prompt": optimized_prompts[0],  # prompt BPO đầu tiên (để so sánh)
        "all_optimized_prompts": unique_prompts,
        "num_clusters": result["num_clusters"],
        "clusters": result["clusters"],
        "cluster_probs": result["cluster_probs"],
        "selected_idx": result["selected_idx"],
        "consensus_scores": result["consensus_scores"]
    }


def main():
    # Load models
    bpo_model, bpo_tokenizer, sbert = load_models(device)

    # Read input
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Process
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Processing prompts"):
            prompt = item.get("prompt", item.get("instruction", ""))

            result = infer_rbpo(
                prompt,
                bpo_model, bpo_tokenizer,
                sbert,
                device=device
            )

            # Ghi kết quả
            output_item = {
                "prompt": prompt,
                "optimized_prompt": result["optimized_prompt"],
                "bpo_prompt": result["bpo_prompt"],
                "num_clusters": result["num_clusters"],
                "clusters": result["clusters"],
                "cluster_probs": result["cluster_probs"],
                "consensus_scores": result["consensus_scores"]
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")

            torch.cuda.empty_cache()

    # Cleanup
    del bpo_model, sbert
    torch.cuda.empty_cache()

    print(f"\n✓ Done! Saved to: {output_jsonl}")

# ============ CONFIG ============
device = "cuda:0"
M = 10  # Số prompt tối ưu cần sinh
distance_threshold = 0.05  # Ngưỡng cosine distance cho clustering
imc_enc = 0.2  # Hệ số khuyến khích cải tiến

# Model paths
bpo_model_path = "THUDM/BPO"
sbert_model_path = "sentence-transformers/all-MiniLM-L12-v2"

# Input/Output
input_jsonl = "testset/vicuna_eval.jsonl"
output_jsonl = "rbpo_results.jsonl"
if __name__ == "__main__":
    main()
