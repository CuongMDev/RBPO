import gc
import os
import shutil
from sentence_transformers import util
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import torch
from config import MODEL_CACHE_PATH
from helper import IMP_ENC, clean_name, experiment_file_name, eval_folder_name, device, M, embedding_models, distance_thresholds, METHOD

print("===== STEP 2: SBERT clustering =====")
torch.cuda.empty_cache()
gc.collect()


def prompt_clustering(key, item, embedding_model, M, distance_threshold):
    ori_prompt = item.get("ori_prompt", "")
    samples = item.get(key, [])
    if M > 0:
        samples = samples[:M]
    
    # Kiểm tra dữ liệu đầu vào
    if len(samples) < 2:
        print(f"Warning: Key '{key}' có ít hơn 2 mẫu ({len(samples)}), bỏ qua clustering")
        return [], None
    
    embeddings = embedding_model.encode(samples, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    
    labels = clustering.fit_predict(embeddings_np)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
        
    clusters = list(clusters.values())
    
    return clusters, embeddings

def representative_selection(item,embedding_model, clusters, embeddings):
    ori_prompt = item.get("ori_prompt", "")
    original_embedding = embedding_model.encode([ori_prompt], convert_to_tensor=True)[0]
    cluster_representatives = []
    single_cluster = False
    
    if len(clusters) == 1:
        cluster = clusters[0]
        c_embeds = torch.stack([embeddings[i] for i in cluster])
        c_sims = util.pytorch_cos_sim(original_embedding, c_embeds)[0]
        c_sorted = torch.argsort(c_sims)
        best_idx = cluster[c_sorted[len(c_sorted) // 2].item()]
        cluster_representatives = [best_idx]
        single_cluster = True
    else: 
        for cluster in clusters:
            if len(cluster) == 1:
                cluster_representatives.append(cluster[0])
            else:
                c_embeds = torch.stack([embeddings[i] for i in cluster])
                c_sims = util.pytorch_cos_sim(original_embedding, c_embeds)[0]
                c_sorted = torch.argsort(c_sims)
                c_median_idx = c_sorted[len(c_sorted) // 2].item()
                cluster_representatives.append(cluster[c_median_idx])
    return cluster_representatives, single_cluster   

def compute_consensus_score(key, item, embedding_model, clusters, embeddings, cluster_representatives, single_cluster):
    if single_cluster:
        return [0.0]
    
    consensus_scores = []
    ori_prompt = item.get("ori_prompt", "")
    original_embedding = embedding_model.encode([ori_prompt], convert_to_tensor=True)[0]
    
    for i, rep_idx in enumerate(cluster_representatives):
        score = 0.0
        rep_embed = embeddings[rep_idx]

        for j, other_rep_idx in enumerate(cluster_representatives):
            if i != j:
                other_embed = embeddings[other_rep_idx]
                sim = util.pytorch_cos_sim(rep_embed, other_embed).item()
                score += sim

        score -= util.pytorch_cos_sim(
            rep_embed, original_embedding
        ).item() * IMP_ENC
        consensus_scores.append(score)
    return consensus_scores

def optimize_prompt_selection(key, item, clusters, embeddings, cluster_representatives, consensus_scores):   
    best_consensus_score_idx = max(range(len(consensus_scores)), key=lambda i: consensus_scores[i])
    best_rep_idx = cluster_representatives[best_consensus_score_idx]
    return best_rep_idx

method_keys = [
    "rbpo_paraphrases",
    "rmepo_paraphrases"
    ]

from helper import GEMMA_EMBEDDING_MODEL

embedding_models = [GEMMA_EMBEDDING_MODEL]
for model_name in embedding_models:
    
    embed_model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=MODEL_CACHE_PATH
    )
    
    distance_threshold = distance_thresholds.get(model_name,None)
    print(f"Using embedding model: {model_name} with distance threshold: {distance_threshold}")
    
    assert distance_threshold is not None, f"Distance threshold not found for embedding model '{model_name}'"
    with open(experiment_file_name, "r") as f:
        lines = f.readlines()

    for line in lines:
        path = line.strip()
        print(f"Processing: {path}")
        with open(f'{eval_folder_name}/{path}.json', "r", encoding="utf-8") as f:
            data = json.load(f)
        # with open(f'{eval_folder_name}/{path}.json', "r") as f:
        #     data = json.load(f)
        print(len(data))
        
        for item in data:
            for key in method_keys:
                clusters, embeddings = prompt_clustering(key, item, embed_model, M, distance_threshold)
                if clusters is None or len(clusters) == 0:  # Bỏ qua nếu dữ liệu không đủ
                    print(f"Warning: Không thể thực hiện clustering cho key '{key}' do dữ liệu không đủ, bỏ qua item này")
                    continue
                cluster_representatives, single_cluster = representative_selection(item, embed_model,clusters, embeddings)
                consensus_scores = compute_consensus_score(key, item, embed_model, clusters, embeddings, cluster_representatives, single_cluster)
                best_rep_idx = optimize_prompt_selection(key, item, clusters, embeddings, cluster_representatives, consensus_scores)
                output_key = "rbpo" if key == "rbpo_paraphrases" else "rmepo"
                item[f'{output_key}_clusters'] = [[item[key][idx] for idx in cluster] for cluster in clusters]
                item[f'{output_key}_prompt'] = item[key][best_rep_idx]
                item[f"{output_key}_cluster_representatives"] = [item[key][idx] for idx in cluster_representatives]
                item[f"{output_key}_consensus_scores"] = consensus_scores
                for key in METHOD:
                    key_to_remove = f"{key}_response"
                    if item.get(key_to_remove) is not None:
                        del item[key_to_remove]
        output_path = f'{eval_folder_name}/{clean_name(model_name)}/{path}.json'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    # remove MODEL_CACHE_PATH
    if os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)