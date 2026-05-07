BATCH_SIZE = 10
import torch

from mepo_inference import MePOModel
from tqdm import tqdm
import json
from helper import evaluation_datasets, evaluator_models, base_llm_models, create_combined_name, mepo_folder_name
from utils import generate_batch

from config import (
    MODEL_CACHE_PATH,
    prompt_template_optimize,
    prompt_template_vicuna
)


def process_mepo_batch(base_llm, data_path, evaluator, model=None, batch_size=BATCH_SIZE):
    """
    Xử lý batch inference MePO cho một bộ kết hợp (base_llm, data_path, evaluator)
    
    Args:
        base_llm: Tên model base LLM
        data_path: Đường dẫn tới file testset JSON
        evaluator: Tên model evaluator
        model: MePOModel instance (nếu None, sẽ tạo mới)
        batch_size: Kích thước batch cho xử lý
        
    Returns:
        result: List chứa các dict với ori_prompt và mepo_prompt
        output_path: Đường dẫn file output
    """
    if model is None:
        model = MePOModel()
    
    result = []
    with open(data_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} samples from {data_path}")
     
    batch_prompts = []
    batch_refs = []

    for sample in tqdm(corpus, desc=f"Processing {base_llm} + {data_path}"):
        ori_prompt = sample.get('instruction') or sample.get('text')
        if not ori_prompt:
            continue

        po_qs_input = model.po_prompt_ins.replace("S_P", ori_prompt)

        batch_prompts.append(po_qs_input)
        batch_refs.append(ori_prompt)

        # chạy khi đủ batch
        if len(batch_prompts) == batch_size:
            outputs = model.generate_batch(batch_prompts)

            for ori, opt in zip(batch_refs, outputs):
                result.append({
                    "ori_prompt": ori,
                    "mepo_prompt": opt
                })

            batch_prompts = []
            batch_refs = []

    # xử lý batch cuối
    if batch_prompts:
        outputs = model.generate_batch(batch_prompts)
        for ori, opt in zip(batch_refs, outputs):
            result.append({
                "ori_prompt": ori,
                "mepo_prompt": opt
            })
    
    return result, base_llm, data_path, evaluator       


def save_mepo_results(result, base_llm, data_path, evaluator):
    """
    Lưu kết quả MePO vào file JSON
    
    Args:
        result: List kết quả từ process_mepo_batch
        base_llm, data_path, evaluator: Thông tin để tạo tên file
    """
    output_path = create_combined_name(base_llm, data_path, evaluator)
    with open(f"{mepo_folder_name}/{output_path}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {mepo_folder_name}/{output_path}.json")


def merge_optim_prompt_with_original(mepo_result, file_path, method_key):
    """
    Gộp MePO result với file JSON gốc (Step 2)
    
    Args:
        mepo_result: List kết quả từ process_mepo_batch ["ori_prompt": ..., "mepo_prompt": ...]
        original_json_path: Đường dẫn tới file JSON gốc (có ori_prompt, bpo_prompt, bpo_res, rbpo_prompt, rbpo_res)
        output_path: Đường dẫn lưu file JSON merged
        
    Returns:
        merged_data: List dữ liệu đã merge
    """
    # Đọc file JSON gốc
    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # Tạo dict từ mepo_result để lookup nhanh
    mepo_map = {}
    for item in mepo_result:
        mepo_map[item["ori_prompt"]] = item[f"{method_key}_prompt"]
    
    # Merge
    merged_data = []
    matched = 0
    for item in original_data:
        ori_prompt = item.get("ori_prompt")
        
        if ori_prompt in mepo_map:
            item[f"{method_key}_prompt"] = mepo_map[ori_prompt]
            matched += 1
        else:
            item[f"{method_key}_prompt"] = None
        
        merged_data.append(item)
    
    # Lưu file merged
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Merged {matched}/{len(merged_data)} items")
    print(f"✓ Saved to {file_path}")
    
    return merged_data

# Doc file_path sau khi đã merge MePO prompt vào (có ori_prompt, mepo_prompt, bpo_prompt, bpo_res, rbpo_prompt, rbpo_res)   
# Chi infer response cho mepo_prompt
# file_path la json

def infer_mepo_res_batch(model,
    tokenizer,
    file_path,
    device='cuda:0',
    is_vicuna=False,
    batch_size=10   
    ):
    print("===== STEP 3: Infer response =====")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Lọc index + prompt hợp lệ
    valid_items = [(i, item["mepo_prompt"]) 
                   for i, item in enumerate(data) 
                   if item.get("mepo_prompt")]
    
    all_indices = [x[0] for x in valid_items]
    all_prompts = [x[1] for x in valid_items]
    
    responses = [None] * len(all_prompts)
    
    # 🚀 Batch inference
    for start in tqdm(range(0, len(all_prompts), batch_size), desc="Batch Infer"):
        batch_prompts = all_prompts[start:start+batch_size]
        if is_vicuna:
            batch_prompts = [prompt_template_vicuna.format(p) for p in batch_prompts]
        
        batch_outputs = generate_batch(
            model,
            tokenizer,
            batch_prompts,
            do_sample=False,
            apply_chat_template=not is_vicuna,
            device=device
        )
        
        responses[start:start+batch_size] = batch_outputs
    
    # Gán lại vào data
    for idx, res in zip(all_indices, responses):
        data[idx]["mepo_response"] = res
    
    # Những item không có prompt
    for item in data:
        if not item.get("mepo_prompt"):
            item["mepo_response"] = None
    
    # Save lại
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Done STEP 3 (BATCH) → {file_path}")


# Main script
if __name__ == "__main__":
    model = MePOModel()
    
    for base_llm in base_llm_models:
        for data_path in evaluation_datasets:
            for evaluator in evaluator_models:
                result, _, _, _ = process_mepo_batch(base_llm, data_path, evaluator, model)
                save_mepo_results(result, base_llm, data_path, evaluator)
