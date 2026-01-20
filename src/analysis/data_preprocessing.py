import json

path = "src/eval_dataset"
folder_name = "vicuna_llama_claude4"
file_path = ["lose_pairwise_results_ori_rbpo.jsonl",
             "lose_pairwise_results_bpo_rbpo.jsonl" ]

# output_path = path + "/" + folder_name +"

"""Chuyen jsonl sang json. ori vs rbpo giu 2 key: prompt_0,prompt_1. rbpo vs bpo giu 3 key: org_prompt,prompt_0,prompt_1."""

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
    print(f"[INFO] Saving preprocessed data to {output_path} ...")
    print(f"[INFO] Total records to save: {len(data)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
for file_name in file_path:
    full_path = f"{path}/{folder_name}/{file_name}"
    data = load_jsonl(full_path)

    unique_data = remove_duplicates(data, file_name)

    output_file_name = file_name.replace(".jsonl", "_preprocessed.json")
    output_full_path = f"{path}/{folder_name}/{output_file_name}"
    save_json(unique_data, output_full_path)


