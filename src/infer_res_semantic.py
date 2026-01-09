"""
Pipeline: Load data -> BPO & RBPO -> Infer responses -> Save

1. Load input data (prompts)
2. Sinh bpo_prompt và rbpo_prompt
3. Infer responses cho: prompt gốc, bpo_prompt, rbpo_prompt
4. Save kết quả vào file
"""

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH, prompt_template_vicuna
from infer_res_rbpo import load_models, infer_bpo, infer_rbpo
from utils import generate_batch

# === CONFIG ===
device = 'cuda:0'

input_jsonl = "optimized_prompts.jsonl"
output_jsonl = "responses_with_semantic.jsonl"
infer_model_path = "lmsys/vicuna-7b-v1.3"

# === LOAD DATA ===
print("Loading input data...")
input_data = [json.loads(l) for l in open(input_jsonl, 'r', encoding='utf-8')]

# === LOAD MODELS ===
print("Loading BPO and SBERT models...")
bpo_model, bpo_tokenizer, sbert = load_models(device)

print("Loading inference model...")
infer_model = AutoModelForCausalLM.from_pretrained(
    infer_model_path,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.float16,
).eval().to(device)

infer_tokenizer = AutoTokenizer.from_pretrained(
    infer_model_path,
    cache_dir=MODEL_CACHE_PATH,
    legacy=False,
)


def infer_responses(prompts, model, tokenizer, device='cuda:0'):
    """Infer responses cho list prompts"""
    formatted = [prompt_template_vicuna.format(p) for p in prompts]
    responses = generate_batch(
        model,
        tokenizer,
        formatted,
        do_sample=False,
        apply_chat_template=False,
        device=device
    )
    return responses


# === MAIN PIPELINE ===
with torch.no_grad():
    with open(output_jsonl, 'w', encoding='utf-8') as fout:

        for item in tqdm(input_data, desc="Processing"):
            prompt = item["prompt"]

            # 1. Sinh BPO prompt
            bpo_result = infer_bpo(prompt, bpo_model, bpo_tokenizer, device=device)
            bpo_prompt = bpo_result["optimized_prompt"]

            # 2. Sinh RBPO prompt
            rbpo_result = infer_rbpo(prompt, bpo_model, bpo_tokenizer, sbert, device=device)
            rbpo_prompt = rbpo_result["optimized_prompt"]

            # 3. Infer responses cho cả 3 prompts (batch)
            all_prompts = [prompt, bpo_prompt, rbpo_prompt]
            all_responses = infer_responses(all_prompts, infer_model, infer_tokenizer, device=device)

            res_original = all_responses[0]
            res_bpo = all_responses[1]
            res_rbpo = all_responses[2]

            # 4. Save kết quả
            out_item = {
                "prompt": prompt,
                "res": res_original,
                "bpo_prompt": bpo_prompt,
                "bpo_res": res_bpo,
                "rbpo_prompt": rbpo_prompt,
                "rbpo_res": res_rbpo,
                # Thêm thông tin RBPO
                "rbpo_num_clusters": rbpo_result["num_clusters"],
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

# Cleanup
del bpo_model, sbert, infer_model
torch.cuda.empty_cache()

print(f"Done! Saved to: {output_jsonl}")
