import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from tqdm import tqdm
from config import prompt_template_optimize
from utils import generate_batch
import gc


def loading_data(input_path, output_path = "optimized_prompts.jsonl", batch_size=10):
    # ---- READ INPUT ----
    data = []
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # ---- INFER BATCH VỚI TQDM ----
    with open(output_path, "w", encoding="utf-8") as f_out:
        for batch_start in tqdm(range(0, len(data), batch_size), desc="Inferring batches"):
            batch_data = data[batch_start:batch_start + batch_size]

            # Chuẩn bị batch prompts
            texts = [item.get("text", item.get("instruction", "")) for item in batch_data]
            for text in texts:
                out = {
                    "prompt": text,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            # Clear cache giữa các batch
            torch.cuda.empty_cache()
            gc.collect()
    print("Done! Saved to:", output_path)
    
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from utils import generate_batch

from config import (
    MODEL_CACHE_PATH,
    prompt_template_optimize,
    prompt_template_vicuna
)

# -----------------------------------------------------
# Helper: generate text
# -----------------------------------------------------

# -----------------------------------------------------
# STEP 1: Generate paraphrase prompts using BPO 
# -----------------------------------------------------
def step1_generate_paraphrase(input_path="optimized_prompts.jsonl",
                            tmp_step1="tmp_step1_r0.jsonl",
                            device='cuda:0',
                            M=10):
    print("\n===== STEP 1: Paraphrase =====")

    optimize_path = "THUDM/BPO"
    model = AutoModelForCausalLM.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, dtype=torch.float16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, use_fast=False, legacy=True
    )
    model.config.return_dict = True

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(tmp_step1, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step1"):
            item = json.loads(line)
            prompt = item["prompt"]

            # Sinh M paraphrases (batch)
            batch_prompts = [prompt_template_optimize.format(prompt) for _ in range(M)]
            paraphrases = generate_batch(
                model, tokenizer,
                batch_prompts,
                temperature=0.9,
                top_p=0.9,
                apply_chat_template=False,
                device=device
            )

            fout.write(json.dumps({
                "prompt": prompt,
                "paraphrase_prompts": paraphrases
            }, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    print("✓ Done Step 1 →", tmp_step1)
    
# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (Vicuna)
# Input  : tmp_step1  (từ Step 1, chứa prompt + paraphrase_prompts)
# Output : tmp_step2  (mỗi item sẽ có thêm paraphrase_responses)
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (dùng hàm generate() có sẵn)
# -----------------------------------------------------
def step2_infer_vicuna(infer_model_path,
    tmp_step1="tmp_step1_r0.jsonl",
    tmp_step2="tmp_step2_r0.jsonl",
    device="cuda:0"):
    print("===== STEP 2: Vicuna inference (full, paraphrases + original prompt) =====")

    # Load model/tokenizer
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
    
    batch_size = 6  # Điều chỉnh tùy theo VRAM

    # File input/output
    with open(tmp_step1, "r", encoding="utf-8") as fin, \
         open(tmp_step2, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step 2 - infer prompts"):
            item = json.loads(line)
            prompt = item["prompt"]
            paraphrases = item["paraphrase_prompts"]

            # === 1) Lọc các prompt unique (bao gồm cả original) ===
            all_prompts = [prompt.strip()] + [p.strip() for p in paraphrases]

            # Tạo mapping: unique_prompt -> index đầu tiên xuất hiện
            unique_prompts = []
            prompt_to_idx = {}  # prompt -> index trong unique_prompts
            original_to_unique = []  # index trong all_prompts -> index trong unique_prompts

            for p in all_prompts:
                if p not in prompt_to_idx:
                    prompt_to_idx[p] = len(unique_prompts)
                    unique_prompts.append(p)
                original_to_unique.append(prompt_to_idx[p])

            # === 2) Infer chỉ các prompt unique ===
            unique_responses = []
            for batch_start in range(0, len(unique_prompts), batch_size):
                batch = unique_prompts[batch_start:batch_start + batch_size]
                batch = [prompt_template_vicuna.format(p) for p in batch]
                # batch = [p for p in batch]

                batch_responses = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    do_sample=False,
                    apply_chat_template=False,
                    device=device
                )
                unique_responses.extend(batch_responses)
                torch.cuda.empty_cache()

            # === 3) Map lại response cho tất cả prompts (bao gồm trùng) ===
            all_responses = [unique_responses[original_to_unique[i]] for i in range(len(all_prompts))]

            # Tách response_original và paraphrase_responses
            response_original = all_responses[0]
            paraphrase_responses = all_responses[1:]

            # === Lưu output STEP 2 ===
            fout.write(json.dumps({
                "prompt": prompt,
                "response_original": response_original,
                "paraphrase_prompts": paraphrases,
                "paraphrase_responses": paraphrase_responses
            }, ensure_ascii=False) + "\n")

    # cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ Done STEP 2 →", tmp_step2)

# -----------------------------------------------------
# STEP 3: SBERT clustering + semantic entropy 
# -----------------------------------------------------
def step3_sbert_clustering(tmp_step2="tmp_step2_r0.jsonl",
    output_jsonl="responses_with_semantic.jsonl",
    distance_threshold=0.05, imp_enc=0.5, M=10,device='cuda:0'):
    """
    STEP 3: SBERT clustering + semantic entropy

    Args:
        device: GPU device
        distance_threshold: Ngưỡng khoảng cách cosine để gộp cụm.
                           Giá trị nhỏ = cụm chặt hơn, nhiều cụm hơn.
                           Giá trị lớn = cụm lỏng hơn, ít cụm hơn.
                           (distance = 1 - cosine_similarity, nên threshold=0.1 ~ similarity=0.9)
    """
    print("===== STEP 3: SBERT clustering + semantic entropy =====")

    sbert = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L12-v2',
        device=device, cache_folder=MODEL_CACHE_PATH)

    with open(tmp_step2, "r", encoding="utf-8") as fin, \
        open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            item = json.loads(line)
            original_prompt = item["prompt"]
            samples = item["paraphrase_prompts"]
            responses = item["paraphrase_responses"]
            if M > 0:
                samples = samples[:M]
                responses = responses[:M]

            # Encode tất cả các câu
            embeddings = sbert.encode(samples, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()

            # Xử lý trường hợp chỉ có 1 sample (không thể clustering)
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

                # Chuyển labels thành clusters (list of lists)
                clusters = {}
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
                clusters = list(clusters.values())

            # ====== 🔥 Chọn câu có similarity MEDIAN với original trong cụm lớn nhất ======
            original_embedding = sbert.encode([original_prompt], convert_to_tensor=True)[0]

            # Nếu chỉ có 1 cụm → chọn phần tử MEDIAN, bỏ qua consensus
            if len(clusters) == 1:
                cluster = clusters[0]
                c_embeds = torch.stack([embeddings[i] for i in cluster])
                c_sims = util.pytorch_cos_sim(original_embedding, c_embeds)[0]
                c_sorted = torch.argsort(c_sims)
                best_idx = cluster[c_sorted[len(c_sorted) // 2].item()]
                cluster_representatives = [best_idx]
                consensus_scores = [0.0]
            else:
                # Thêm cụm chứa câu gốc (index = -1, dùng original_embedding)
                # Để có >= 3 cụm cho consensus score

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

                # Thêm cụm gốc: đại diện = -1 (đặc biệt, dùng original_embedding)
                # clusters.append([-1])  # cụm chỉ chứa câu gốc
                # cluster_representatives.append(-1)

                # ====== Cross-cluster consensus score ======
                # Tính score cho mỗi đại diện = tổng(sim với đại diện khác × kích thước cụm đó)
                consensus_scores = []
                for i, rep_idx in enumerate(cluster_representatives):
                    score = 0.0
                    # Lấy embedding của đại diện i
                    if rep_idx == -1:
                        rep_embed = original_embedding
                    else:
                        rep_embed = embeddings[rep_idx]

                    for j, other_rep_idx in enumerate(cluster_representatives):
                        if i != j:
                            # Lấy embedding của đại diện j
                            if other_rep_idx == -1:
                                other_embed = original_embedding
                            else:
                                other_embed = embeddings[other_rep_idx]

                            sim = util.pytorch_cos_sim(rep_embed, other_embed).item()
                            # weight = len(clusters[j])  # trọng số = kích thước cụm
                            score += sim

                    score -= util.pytorch_cos_sim(rep_embed, original_embedding).item() * imp_enc  # khuyến khích cải tiến
                    consensus_scores.append(score)

                # print(consensus_scores)

                # Chọn đại diện có consensus score cao nhất (bỏ qua cụm gốc -1)
                # Chỉ xét các cụm có index thực (không phải -1)
                valid_indices = [i for i, rep in enumerate(cluster_representatives) if rep != -1]
                best_consensus_idx = max(valid_indices, key=lambda i: consensus_scores[i])

                # Sau khi có cụm tốt nhất, chọn phần tử MEDIAN trong cụm đó
                best_cluster = clusters[best_consensus_idx]
                if len(best_cluster) == 1:
                    best_idx = best_cluster[0]
                else:
                    # Tính similarity với original cho từng phần tử trong cụm
                    bc_embeds = torch.stack([embeddings[i] for i in best_cluster])
                    bc_sims = util.pytorch_cos_sim(original_embedding, bc_embeds)[0]
                    # Chọn phần tử có similarity MEDIAN
                    bc_sorted = torch.argsort(bc_sims)
                    best_idx = best_cluster[bc_sorted[len(bc_sorted) // 2].item()]

                # Xóa cụm gốc khỏi clusters để output đúng
                # clusters = clusters[:-1]
                # cluster_representatives = cluster_representatives[:-1]
                # consensus_scores = consensus_scores[:-1]

            # Cluster probabilities
            cluster_probs = [len(c)/len(samples) for c in clusters]

            # Semantic entropy
            entropy = -sum(p * (math.log(p) if p > 0 else 0) for p in cluster_probs)

            # Confidence score
            K = len(clusters)
            conf_score = 1 - (entropy / math.log(K)) if K > 1 else 1.0

            # ==== Ghi ra JSONL ====
            fout.write(json.dumps({
                "prompt": original_prompt,
                "response_original": item["response_original"],
                "bpo_prompt": samples[0],
                "bpo_response": responses[0],
                "optimized_prompt": samples[best_idx],
                "optimized_response": responses[best_idx],
                "paraphrase_responses": responses,
                "paraphrase_prompts": samples,
                "clusters": clusters,
                "cluster_representatives": cluster_representatives,
                "consensus_scores": consensus_scores,
                "cluster_probs": cluster_probs,
                "semantic_entropy": entropy,
                "conf_score": conf_score
            }, ensure_ascii=False) + "\n")

        print("✓ Done STEP 3 (SBERT)")
 
import torch
import json
import os
import time
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from ranking_utils import (
    load_ranking_prompt,
    read_jsonl,
    run_ranking_loop,
    run_followup_inference
)

# =========================
# Claude wrapper
# =========================
def run_with_claude(client,evaluator, prompt, model, tokenizer, device, retries=3):
    """
    Call Claude for reasoning, then use local model to extract boxed answer
    """
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=evaluator,
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )

            reasoning = response.choices[0].message.content

            result = run_followup_inference(
                reasoning,
                model,
                tokenizer,
                device
            )

            torch.cuda.empty_cache()
            return result

        except Exception:
            time.sleep(2)

    raise RuntimeError("Claude API failed after retries")


# =========================
# Main ranking pipeline
# =========================
def run_pairwise_ranking(
    evaluator,
    input_jsonl="responses_with_semantic.jsonl",
    output_jsonls=None,
    output_result="results.json",
    model_name="Qwen/Qwen3-4B",
    device="cuda:0",
    api_key_env="OPENROUTER_API_KEY"
):
    """
    Runs pairwise ranking:
      - Original vs BPO
      - Original vs RBPO
      - BPO vs RBPO
    Outputs:
      - Per-pair JSONL logs
      - Summary stats JSON
    """

    if output_jsonls is None:
        output_jsonls = [
            "lose_pairwise_results_ori_bpo.jsonl",
            "lose_pairwise_results_ori_rbpo.jsonl",
            "lose_pairwise_results_bpo_rbpo.jsonl"
        ]

    # ---------------------
    # Setup client
    # ---------------------
    api_key = os.environ.get(api_key_env)
    if api_key is None:
        raise RuntimeError(f"Missing API key in env: {api_key_env}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # ---------------------
    # Load model
    # ---------------------
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODEL_CACHE_PATH
    )

    # ---------------------
    # Load data
    # ---------------------
    raw_prompt = load_ranking_prompt()
    rows = read_jsonl(input_jsonl)

    stats = {}

    # ---------------------
    # Helper eval fn
    # ---------------------
    def eval_fn(prompt):
        return run_with_claude(
            client,
            evaluator,
            prompt,
            model,
            tokenizer,
            device
        )

    # ===== 0. ORIGINAL vs BPO =====
    orig_0, bpo_0, draw_0 = run_ranking_loop(
        rows=rows,
        eval_fn=eval_fn,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonls[0],
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["prompt"],
        get_output_1_fn=lambda item: item["response_original"],
        get_prompt_2_fn=lambda item: item["bpo_prompt"],
        get_output_2_fn=lambda item: item["bpo_response"],
        label_0="original_win",
        label_1="bpo_win",
        label_2="draw",
        save_winner_0=False,
        save_winner_1=False,
        handle_bias=False
    )

    stats["original_vs_bpo"] = {
        "original_win": orig_0,
        "bpo_win": bpo_0,
        "draw": draw_0,
    }

    # ===== 1. ORIGINAL vs RBPO =====
    orig_1, rbpo_1, draw_1 = run_ranking_loop(
        rows=rows,
        eval_fn=eval_fn,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonls[1],
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["prompt"],
        get_output_1_fn=lambda item: item["response_original"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_2_fn=lambda item: item["optimized_response"],
        label_0="original_win",
        label_1="rbpo_win",
        label_2="draw",
        save_winner_0=False,
        save_winner_1=False,
        handle_bias=False
    )

    stats["original_vs_rbpo"] = {
        "original_win": orig_1,
        "rbpo_win": rbpo_1,
        "draw": draw_1,
    }

    # ===== 2. BPO vs RBPO =====
    bpo_2, rbpo_2, draw_2 = run_ranking_loop(
        rows=rows,
        eval_fn=eval_fn,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonls[2],
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["bpo_prompt"],
        get_output_1_fn=lambda item: item["bpo_response"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_2_fn=lambda item: item["optimized_response"],
        label_0="bpo_win",
        label_1="rbpo_win",
        label_2="draw",
        save_winner_0=False,
        save_winner_1=False,
        handle_bias=False
    )

    stats["bpo_vs_rbpo"] = {
        "bpo_win": bpo_2,
        "rbpo_win": rbpo_2,
        "draw": draw_2,
    }

    # ---------------------
    # Save results
    # ---------------------
    with open(output_result, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved ranking results to {output_result}")
    return stats
