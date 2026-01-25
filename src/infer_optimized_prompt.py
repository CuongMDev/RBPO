import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from tqdm import tqdm
from config import prompt_template_optimize
from utils import generate_batch

model_path = 'THUDM/BPO'

input_jsonl = "testset/demo.json"
output_jsonl = "optimized_prompts.jsonl"

device = 'cuda:0'
batch_size = 10  # Điều chỉnh tùy theo VRAM

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH, use_fast=False)
model.config.return_dict = True

# Nếu pad_token chưa set, set = eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- READ INPUT ----
data = []
if input_jsonl.endswith(".jsonl"):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
elif input_jsonl.endswith(".json"):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        data = json.load(f)

# ---- INFER BATCH VỚI TQDM ----
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Inferring batches"):
        batch_data = data[batch_start:batch_start + batch_size]

        # Chuẩn bị batch prompts
        texts = [item.get("text", item.get("instruction", "")) for item in batch_data]
        prompts = [prompt_template_optimize.format(text) for text in texts]

        # Generate batch
        # outputs = generate_batch(
        #     model,
        #     tokenizer,
        #     prompts,
        #     max_new_tokens=1024,
        #     do_sample=True,
        #     top_p=0.9,
        #     temperature=0.6,
        #     apply_chat_template=False,
        #     device=device
        # )

        # Save từng kết quả
        # for text, output in zip(texts, outputs):
        for text in texts:
            out = {
                "prompt": text,
                # "optimized_prompt": output
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Clear cache giữa các batch
        torch.cuda.empty_cache()

print("Done! Saved to:", output_jsonl)
