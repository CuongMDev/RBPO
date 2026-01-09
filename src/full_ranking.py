import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_PATH
from ranking_utils import (
    load_ranking_prompt,
    rank_pair,
    run_eval_model
)

device = "cuda:0"
model_name = "unsloth/gemma-3-27b-it-bnb-4bit"

# ==== FILES ====
input_file = "responses_with_semantic.jsonl"
output_file = "all_ranking_results.jsonl"

# ==== LOAD RANKING PROMPT TEMPLATE ====
RAW_PROMPT = load_ranking_prompt("ranking_prompt.txt")

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH)

# ==== MAIN RANKING ====
if __name__ == "__main__":
    eval_fn = lambda prompt: run_eval_model(prompt, model, tokenizer, device)

    with open(output_file, "w", encoding="utf-8") as fout:
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Ranking all paraphrases"):
                item = json.loads(line)

                prompt = item["prompt"]
                original = item["response_original"]
                samples = item["paraphrase_responses"]

                # === 1) Lọc unique samples ===
                unique_samples = []
                sample_to_idx = {}
                original_to_unique = []

                for para in samples:
                    para_stripped = para.strip()
                    if para_stripped not in sample_to_idx:
                        sample_to_idx[para_stripped] = len(unique_samples)
                        unique_samples.append(para_stripped)
                    original_to_unique.append(sample_to_idx[para_stripped])

                # === 2) Ranking chỉ các unique samples ===
                unique_winners = []
                for para in unique_samples:
                    if para == original.strip():
                        unique_winners.append(2)
                        continue

                    w = rank_pair(eval_fn, RAW_PROMPT, prompt, para, original)
                    unique_winners.append(w)

                # === 3) Map lại winners cho tất cả samples ===
                winners = [unique_winners[original_to_unique[i]] for i in range(len(samples))]

                fout.write(json.dumps(winners, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"DONE → {output_file}")
