import torch, os, gc, json

from tqdm import tqdm

from helper import device, experiment_file_name, M, eval_folder_name
from config import MODEL_CACHE_PATH, prompt_template_optimize
from inference_batch import merge_mepo_with_original
from utils import generate_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mepo_inference import MePOModel

torch.manual_seed(42)
# =========================
# Step 1: Infer MePO prompt for all items
# =========================
mepo_model = MePOModel()
batch_size = 8

result = []

with open(experiment_file_name, "r") as f:
    lines = f.readlines()
    
for line in lines:
    path = line.strip()
    print(f"Processing: {path}")
    with open(f'{eval_folder_name}/{path}.json', "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {path}")

    batch_prompts = []
    batch_refs = []
    
    for sample in tqdm(data, desc=f"Processing {path}"):
        ori_prompt = sample.get("ori_prompt", "")
        po_qs_input = mepo_model.po_prompt_ins.replace("S_P", ori_prompt)

        batch_prompts.append(po_qs_input)
        batch_refs.append(ori_prompt)

        # chạy khi đủ batch
        if len(batch_prompts) == batch_size:
            outputs = mepo_model.generate_batch(batch_prompts)

            for ori, opt in zip(batch_refs, outputs):
                result.append({
                    "ori_prompt": ori,
                    "mepo_prompt": opt
                })
            batch_prompts = []
            batch_refs = []

    # xử lý batch cuối
    if batch_prompts:
        outputs = mepo_model.generate_batch(batch_prompts)
        for ori, opt in zip(batch_refs, outputs):
            result.append({
                "ori_prompt": ori,
                "mepo_prompt": opt
            })

    merge_mepo_with_original(result, f'{path}.json')

# =========================
# Step 2: Infer MePO M candidates prompt for all items
# =========================

with open(experiment_file_name, "r") as f:
    lines = f.readlines()
for line in lines:
    path = line.strip()
    print(f"Processing: {path}")
    with open(f'{eval_folder_name}/{path}.json', "r") as f:
        data = json.load(f)
    
    batch_prompts = []
    batch_refs = []
    
    for sample in tqdm(data, desc=f"Processing {path}"):
        ori_prompt = sample.get("ori_prompt", "")

        batch_prompts = [
            prompt_template_optimize.format(ori_prompt) for _ in range(M)
        ]
        
        mepo_paraphrases = mepo_model.generate_paraphrase_batch(batch_prompts)
        
        sample["mepo_paraphrases"] = mepo_paraphrases
    
    with open(f'{eval_folder_name}/{path}.json', "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
print("Delete MePO model to free up memory")
mepo_model.clean_up()