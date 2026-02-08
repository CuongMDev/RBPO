import json
import os

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
def filter_keys(data, keep_keys):
    """
    Giữ lại chỉ các key nằm trong keep_keys cho mỗi item.
    """
    filtered = []
    for item in data:
        filtered.append({k: item[k] for k in keep_keys if k in item})
    return filtered

def rename_keys(data, key_map):
    """
    Đổi tên key trong mỗi item theo key_map dạng:
    [(old_key1, new_key1), (old_key2, new_key2), ...]
    """
    rename_dict = dict(key_map)
    renamed = []

    for item in data:
        new_item = {}
        for k, v in item.items():
            if k in rename_dict:
                new_item[rename_dict[k]] = v
            else:
                new_item[k] = v
        renamed.append(new_item)

    return renamed

def data_preprocessing(input_jsonl_path: str,
    output_json_path: str,
    keep_keys=None,
    key_map=None):
    data = load_jsonl(input_jsonl_path)

    # unique_data = remove_duplicates(data, os.path.basename(input_jsonl_path))
    
    if keep_keys is not None:
        data = filter_keys(data, keep_keys)

    if key_map is not None:
        data = rename_keys(data, key_map)

    data = add_id(data, start_id=1)

    save_json(data, output_json_path)