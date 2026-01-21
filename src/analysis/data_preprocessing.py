import json
import os

# path = "src/analysis"

# folder_names = ["claude4/vicuna_llama_claude4",
#                "claude4/vicuna_vicuna_claude4",
#                "claude4/dolly_vicuna_claude4",
#                "claude4/dolly_llama_claude4" ]

# file_paths = ["lose_pairwise_results_ori_rbpo.jsonl",
#              "lose_pairwise_results_bpo_rbpo.jsonl" ]

"""Chuyển jsonl sang json. 
ori vs rbpo giữ 2 key: prompt_0, prompt_1. 
rbpo vs bpo giữ 3 key: org_prompt, prompt_0, prompt_1.
"""

# 1. Loại trùng lặp các bản ghi jsonl
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def remove_duplicates(data, file_name):
    """
    - ori_rbpo  : giữ (prompt_0, prompt_1)
    - bpo_rbpo  : giữ (org_prompt, prompt_0, prompt_1)
    Loại trùng dựa trên tuple các key này.
    """
    seen = set()
    unique_data = []

    is_ori_rbpo = "ori_rbpo" in file_name

    for item in data:
        if is_ori_rbpo:
            key_tuple = (
                item.get("prompt_0"),
                item.get("prompt_1"),
            )
            cleaned_item = {
                "prompt_0": item.get("prompt_0"),
                "prompt_1": item.get("prompt_1"),
                "winner": item.get("winner"),
            }
        else:
            key_tuple = (
                item.get("org_prompt"),
                item.get("prompt_0"),
                item.get("prompt_1"),
            )
            cleaned_item = {
                "org_prompt": item.get("org_prompt"),
                "prompt_0": item.get("prompt_0"),
                "prompt_1": item.get("prompt_1"),
                "winner": item.get("winner"),
            }

        if key_tuple not in seen:
            seen.add(key_tuple)
            unique_data.append(cleaned_item)

    return unique_data


def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[INFO] Saving preprocessed data to {output_path} ...")
    print(f"[INFO] Total records to save: {len(data)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def add_id(data, start_id=1):
    """
    Gán id tuần tự cho mỗi bản ghi sau khi preprocess.
    """
    new_data = []
    for idx, item in enumerate(data, start=start_id):
        item_with_id = {"id": idx}
        item_with_id.update(item)
        new_data.append(item_with_id)
    return new_data

    
# ================== MAIN LOOP ==================
def data_preprocessing(folder_names, file_paths, path):
    for folder in folder_names:
        for file_name in file_paths:
            full_path = os.path.join(path, folder, file_name)

            if not os.path.exists(full_path):
                print(f"[WARN] File not found: {full_path}")
                continue

            print(f"[INFO] Processing: {full_path}")

            data = load_jsonl(full_path)
            unique_data = remove_duplicates(data, file_name)

            # reset id cho từng file
            unique_data = add_id(unique_data, start_id=1)

            output_file_name = file_name.replace(".jsonl", "_preprocessed.json")
            output_full_path = os.path.join(path, folder, output_file_name)

            save_json(unique_data, output_full_path)


