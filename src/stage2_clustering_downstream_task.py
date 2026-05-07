import gc
import os
import shutil
from sentence_transformers import util
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import torch
from config import MODEL_CACHE_PATH
from helper import (IMP_ENC, MINILM_EMBEDDING_MODEL, PIQA, RMEPO, clean_name, experiment_file_name,
    eval_folder_name, device, M, embedding_models, distance_thresholds, METHOD, 
    GEMMA_EMBEDDING_MODEL, downstream_folder_name, downstream_tasks, RBPO)

def prompt_clustering(key,
    item,
    embedding_model,
    M,
    distance_threshold,
    ori_prompt_key
):
    ori_prompt = item.get(ori_prompt_key, "")
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

def representative_selection(item,embedding_model, clusters, embeddings, ori_prompt_key):
    ori_prompt = item.get(ori_prompt_key, "")
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

def clustering_and_selection(path, key, embedding_model, M, distance_threshold, ori_prompt_key):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        clusters, embeddings = prompt_clustering(f"{key}_rephrases", item, embedding_model, M, distance_threshold, ori_prompt_key)
        if clusters is None or len(clusters) == 0:  # Bỏ qua nếu dữ liệu không đủ
            print(f"Warning: Không thể thực hiện clustering cho key '{key}' do dữ liệu không đủ, bỏ qua item này")
            return None
        cluster_representatives, single_cluster = representative_selection(item, embedding_model,clusters, embeddings, ori_prompt_key)
        consensus_scores = compute_consensus_score(key, item, embedding_model, clusters, embeddings, cluster_representatives, single_cluster)
        best_rep_idx = optimize_prompt_selection(key, item, clusters, embeddings, cluster_representatives, consensus_scores)
        item[f'{key}_clusters'] = [[item[f"{key}_rephrases"][idx] for idx in cluster] for cluster in clusters]
        item[f'{key}_prompt'] = item[f"{key}_rephrases"][best_rep_idx]
        item[f"{key}_cluster_representatives"] = [item[f"{key}_rephrases"][idx] for idx in cluster_representatives]
        item[f"{key}_consensus_scores"] = consensus_scores
    
    return data



folder_name = downstream_folder_name
embedding_models = [
    GEMMA_EMBEDDING_MODEL,
    MINILM_EMBEDDING_MODEL
]
METHOD = [
    RBPO,
    RMEPO
]

for model_name in embedding_models:
    embed_model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=MODEL_CACHE_PATH
    )
    
    distance_threshold = distance_thresholds.get(model_name,None)
    print(f"Using embedding model: {model_name} with distance threshold: {distance_threshold}")
    
    assert distance_threshold is not None, f"Distance threshold not found for embedding model '{model_name}'"
    for method_key in METHOD:
        for task in downstream_tasks:
            save_path = None
            objs = None
            import torch, gc
            print('='*80)
            print(torch.cuda.empty_cache(),gc.collect())
            print(f"\nProcessing task: {task} with method: {method_key}")
            
            if task == "demo":
                continue
            elif task == PIQA:
                input_path = f"{downstream_folder_name}/{method_key}/{task}_optim.json"
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                objs = clustering_and_selection(input_path, method_key, embed_model, M, distance_threshold, ori_prompt_key="raw_question")
                save_path = f"{downstream_folder_name}/{method_key}/{clean_name(model_name)}/{task}.json"
                if not os.path.exists(save_path):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(objs, f, ensure_ascii=False, indent=2)
                
            # if task.upper() == 'BBH':
            #     multiple_choice = [
            #         # ARC_C,
            #         # ARC_E,
            #         'snarks',
            #         'ruin_names', 
            #         'hyperbaton', 
            #         'movie_recommendation',
            #         'penguins_in_a_table',
            #         'temporal_sequences',       
            #         'date_understanding',
            #         'disambiguation_qa', 
            #         'logical_deduction_three_objects',
            #         'logical_deduction_five_objects',
            #         'logical_deduction_seven_objects',
            #         'reasoning_about_colored_objects',
            #         'salient_translation_error_detection',
            #         'tracking_shuffled_objects_five_objects', 
            #         'tracking_shuffled_objects_seven_objects',
            #         'tracking_shuffled_objects_three_objects'
            #     ] # NOTE: select between Multiple choices, ARC
            #     # [
            #     #     'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
            #     #     'logical_deduction_seven_objects', 'logical_deduction_three_objects',
            #     #     'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
            #     #     'ruin_names', 'salient_translation_error_detection', 'snarks',
            #     #     'temporal_sequences', 'tracking_shuffled_objects_five_objects',
            #     #     'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
            #     #     'causal_judgement','formal_fallacies','navigate','web_of_lies',
            #     #     'sports_understanding','boolean_expressions',
            #     #     'multistep_arithmetic_two', 'object_counting', 'word_sorting'
            #     # ]
            #     for subtask in multiple_choice:
            #         input_path = f"{downstream_folder_name}/{method_key}/{task}/{subtask}_optim.json"
            #         objs = clustering_and_selection(input_path, method_key, embed_model, M, distance_threshold, ori_prompt_key="raw_question")
            #         save_path = f"{downstream_folder_name}/{method_key}/{clean_name(model_name)}/{task}/{subtask}_cluster.json"
            #         if not os.path.exists(save_path):
            #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #         with open(save_path, "w", encoding="utf-8") as f:
            #             json.dump(objs, f, ensure_ascii=False, indent=2)
                    
            # else:
            #     input_path = f"{downstream_folder_name}/{method_key}/{task}_optim.json"
            #     with open(input_path, 'r', encoding='utf-8') as f:
            #         data = json.load(f)
            #     objs = clustering_and_selection(input_path, method_key, embed_model, M, distance_threshold, ori_prompt_key="raw_question")
            #     save_path = f"{downstream_folder_name}/{method_key}/{clean_name(model_name)}/{task}_cluster.json"
            #     if not os.path.exists(save_path):
            #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     with open(save_path, "w", encoding="utf-8") as f:
            #         json.dump(objs, f, ensure_ascii=False, indent=2)
                    
    del embed_model
    torch.cuda.empty_cache()
    gc.collect()                    
    if os.path.exists(MODEL_CACHE_PATH):
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)