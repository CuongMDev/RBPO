import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from utils import generate_batch
from semantic import step1_generate_paraphrase
import gc

from config import (
    MODEL_CACHE_PATH,
    prompt_template_vicuna
)

torch.manual_seed(42)

# ============ CONFIG ============
device = "cuda:0"
tmp_step1 = "tmp_step1_r0.jsonl"
tmp_step2 = "tmp_step2_r0.jsonl"
output_jsonl = "responses_with_semantic.jsonl"
infer_model_path = "meta-llama/Llama-2-7b-chat-hf"
M = 10

# Load model
model = AutoModelForCausalLM.from_pretrained(
    infer_model_path,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.float16
).eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(
    infer_model_path,
    cache_dir=MODEL_CACHE_PATH,
    legacy=False
)


# -----------------------------------------------------
# STEP 2: SBERT clustering + chọn optimized_prompt
# Input  : tmp_step1 (chứa prompt + paraphrase_prompts)
# Output : tmp_step2 (chứa optimized_prompt đã chọn)
# -----------------------------------------------------
def step2_sbert_clustering(device='cuda:0', distance_threshold=0.05, imp_enc=0.5):
    """
    STEP 2: SBERT clustering để chọn optimized_prompt
    """
    print("===== STEP 2: SBERT clustering =====")
    torch.cuda.empty_cache()
    gc.collect()

    sbert = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L12-v2',
        device=device, cache_folder=MODEL_CACHE_PATH)

    with open(tmp_step1, "r", encoding="utf-8") as fin, \
        open(tmp_step2, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step 2 - SBERT clustering"):
            item = json.loads(line)
            original_prompt = item["prompt"]
            samples = item["paraphrase_prompts"]
            if M > 0:
                samples = samples[:M]

            # Encode tất cả các câu
            embeddings = sbert.encode(samples, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()

            # Xử lý trường hợp chỉ có 1 sample
            if len(samples) == 1:
                clusters = [[0]]
            else:
                # AgglomerativeClustering với cosine distance
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings_np)

                # Chuyển labels thành clusters
                clusters = {}
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
                clusters = list(clusters.values())

            # Chọn câu có similarity MEDIAN với original trong cụm lớn nhất
            original_embedding = sbert.encode([original_prompt], convert_to_tensor=True)[0]

            # Nếu chỉ có 1 cụm → chọn phần tử MEDIAN
            if len(clusters) == 1:
                cluster = clusters[0]
                c_embeds = torch.stack([embeddings[i] for i in cluster])
                c_sims = util.pytorch_cos_sim(original_embedding, c_embeds)[0]
                c_sorted = torch.argsort(c_sims)
                best_idx = cluster[c_sorted[len(c_sorted) // 2].item()]
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

                    score -= util.pytorch_cos_sim(rep_embed, original_embedding).item() * imp_enc
                    consensus_scores.append(score)

                # Chọn đại diện có consensus score cao nhất
                best_consensus_idx = max(range(len(consensus_scores)), key=lambda i: consensus_scores[i])

                # Chọn phần tử MEDIAN trong cụm tốt nhất
                best_cluster = clusters[best_consensus_idx]
                if len(best_cluster) == 1:
                    best_idx = best_cluster[0]
                else:
                    bc_embeds = torch.stack([embeddings[i] for i in best_cluster])
                    bc_sims = util.pytorch_cos_sim(original_embedding, bc_embeds)[0]
                    bc_sorted = torch.argsort(bc_sims)
                    best_idx = best_cluster[bc_sorted[len(bc_sorted) // 2].item()]

            # Cluster probabilities
            cluster_probs = [len(c)/len(samples) for c in clusters]

            # Semantic entropy
            entropy = -sum(p * (math.log(p) if p > 0 else 0) for p in cluster_probs)

            # Confidence score
            K = len(clusters)
            conf_score = 1 - (entropy / math.log(K)) if K > 1 else 1.0

            # Ghi ra JSONL
            fout.write(json.dumps({
                "prompt": original_prompt,
                "bpo_prompt": samples[0],
                "optimized_prompt": samples[best_idx],
                "paraphrase_prompts": samples,
                "clusters": clusters,
                "cluster_representatives": cluster_representatives,
                "consensus_scores": consensus_scores,
                "cluster_probs": cluster_probs,
                "semantic_entropy": entropy,
                "conf_score": conf_score
            }, ensure_ascii=False) + "\n")
    
    print("✓ Done STEP 2 →", tmp_step2)
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------
# STEP 3: Infer response cho optimized_prompt đã chọn
# Input  : tmp_step2 (chứa optimized_prompt)
# Output : output_jsonl (thêm optimized_res)
# -----------------------------------------------------
def step3_infer_response(device="cuda:0"):
    """
    STEP 3: Sinh response cho prompt gốc và optimized_prompt
    """
    print("===== STEP 3: Infer response =====")

    with open(tmp_step2, "r", encoding="utf-8") as fin, \
        open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step 3 - Infer response"):
            item = json.loads(line)
            original_prompt = item["prompt"].strip()
            bpo_prompt = item["bpo_prompt"].strip()
            optimized_prompt = item["optimized_prompt"].strip()

            # Lọc unique prompts để tránh infer trùng
            all_prompts = [original_prompt, bpo_prompt, optimized_prompt]
            unique_prompts = []
            prompt_to_idx = {}
            original_to_unique = []

            for p in all_prompts:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(unique_prompts)
                    unique_prompts.append(p)
                original_to_unique.append(prompt_to_idx[p])

            # unique_prompts = [prompt_template_vicuna.format(p) for p in unique_prompts] # for Vicuna-style model (turn off apply_chat_template)

            # Sinh response chỉ cho unique prompts
            unique_responses = generate_batch(
                model,
                tokenizer,
                unique_prompts,
                do_sample=False,
                apply_chat_template=True,
                device=device
            )

            # Map lại response cho tất cả prompts
            responses = [unique_responses[original_to_unique[i]] for i in range(len(all_prompts))]

            # Thêm response vào item
            item["response_original"] = responses[0]
            item["bpo_response"] = responses[1]
            item["optimized_response"] = responses[2]

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ Done STEP 3 →", output_jsonl)


# -----------------------------------------------------
# RUN ALL STEPS
# -----------------------------------------------------
if __name__ == "__main__":
    # step1_generate_paraphrase()
    # step2_sbert_clustering()
    step3_infer_response()
    print("\n🎉 ALL DONE!")
