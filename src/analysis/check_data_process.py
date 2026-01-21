import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_prompt0_alignment(file_a, file_b):
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    len_a = len(data_a)
    len_b = len(data_b)

    if len_a != len_b:
        print(f"[WARN] Length mismatch: file A = {len_a}, file B = {len_b}")

    min_len = min(len_a, len_b)
    mismatches = []

    for i in range(min_len):
        a = data_a[i]
        b = data_b[i]

        p0_a = a.get("prompt_0")
        p0_b = b.get("prompt_0")

        if p0_a != p0_b:
            mismatches.append({
                "row": i,
                "id_a": a.get("id"),
                "id_b": b.get("id"),
                "prompt_0_a": p0_a,
                "prompt_0_b": p0_b
            })

    return mismatches


if __name__ == "__main__":
    file_a = "D:\\Folder F\\phamtuananh@23020010\\UET.iSEML\\Reliable Black-Box Prompt Optimization\\src\\analysis\\claude4\\vicuna_llama_claude4\\lose_pairwise_results_ori_rbpo_preprocessed.json"
    file_b = "D:\\Folder F\\phamtuananh@23020010\\UET.iSEML\\Reliable Black-Box Prompt Optimization\\src\\analysis\\claude4\\vicuna_vicuna_claude4\\lose_pairwise_results_ori_rbpo_preprocessed.json"

    mismatches = check_prompt0_alignment(file_a, file_b)

    if not mismatches:
        print("OK: Tất cả prompt_0 đều khớp theo thứ tự.")
    else:
        print(f"FOUND {len(mismatches)} mismatches:\n")
        for x in mismatches:
            print("-" * 80)
            print(f"Row: {x['row']}")
            print(f"  id_a: {x['id_a']}, id_b: {x['id_b']}")
            print(f"  prompt_0_a: {x['prompt_0_a']}")
            print(f"  prompt_0_b: {x['prompt_0_b']}")
