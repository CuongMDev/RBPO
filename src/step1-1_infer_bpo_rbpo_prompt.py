from helper import BPO_MODEL, device, experiment_file_name, M, eval_folder_name
from config import MODEL_CACHE_PATH, prompt_template_optimize
import torch, os, gc, json
from utils import generate, generate_batch


from transformers import AutoModelForCausalLM, AutoTokenizer

experiment_file_name = "demo_experiment.txt"

torch.manual_seed(42)
# Loading BPO
print("Loading BPO model once...")
bpo_model = AutoModelForCausalLM.from_pretrained(
    BPO_MODEL,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.float16
).eval().to(device)

bpo_tokenizer = AutoTokenizer.from_pretrained(
    BPO_MODEL,
    cache_dir=MODEL_CACHE_PATH,
    use_fast=False,
    legacy=True
)
bpo_model.config.return_dict = True
if bpo_tokenizer.pad_token_id is None:
    bpo_tokenizer.pad_token_id = bpo_tokenizer.eos_token_id


# Doc tung hang trong experiment.txt co cac path. Moi lan doc thi toi se vao cac path day de chinh sua noi dung
# ========================================
# Step 1: Infer BPO prompt for all items
# ========================================
with open(experiment_file_name, "r") as f:
    lines = f.readlines()

for line in lines:
    path = line.strip()
    print(f"Processing: {path}")
    with open(f'{eval_folder_name}/{path}.json', "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data:
        assert "ori_prompt" in item, f"Missing 'ori_prompt' in item: {item}"
        ori_prompt = item.get("ori_prompt", "")
        
        prompt = prompt_template_optimize.format(ori_prompt) # format template
        
        item["bpo_prompt"] = generate(
            bpo_model,
            bpo_tokenizer,
            prompt,
            temperature=0.9,
            top_p=0.9,
            apply_chat_template=False,
            device=device,
        )
    
    with open(f'{eval_folder_name}/{path}.json', "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  
        
# =========================
# Step 2: Infer RBPO M candidates prompt for all items
# =========================
with open(experiment_file_name, "r") as f:
    lines = f.readlines()

for line in lines:
    path = line.strip()
    print(f"Processing: {path}")
    with open(f'{eval_folder_name}/{path}.json', "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        assert "ori_prompt" in item, f"Missing 'ori_prompt' in item: {item}"
        ori_prompt = item.get("ori_prompt", "")
        
        batch_prompt = [prompt_template_optimize.format(ori_prompt) for _ in range(M)]
        rbpo_paraphrases = generate_batch(
            bpo_model,
            bpo_tokenizer,
            batch_prompt,
            temperature=0.9,
            top_p=0.9,
            apply_chat_template=False,
            device=device,  
        )
        assert len(batch_prompt) == len(rbpo_paraphrases), "Mismatch between batch size and generated paraphrases"
        item["rbpo_paraphrases"] = rbpo_paraphrases
    
    with open(f'{eval_folder_name}/{path}.json', "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)      
        
print("Delete BPO model and tokenizer to free up memory")