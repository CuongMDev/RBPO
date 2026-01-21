import json
import os

def jsonl_to_sorted_json(input_jsonl_path: str,
                          output_json_path: str,
                          sort_key: str = "id"):

    if not os.path.exists(input_jsonl_path):
        print(f"[SKIP] Input file not found: {input_jsonl_path}")
        return

    data = []

    # ---------- 1. Load JSONL ----------
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {i} JSON decode error: {e}")

    if not data:
        print(f"[SKIP] No valid rows in: {input_jsonl_path}")
        return

    # ---------- 2. Sort ----------
    try:
        data.sort(key=lambda x: x.get(sort_key, float("inf")))
    except TypeError:
        data.sort(key=lambda x: str(x.get(sort_key, "")))

    # ---------- 3. Ghi JSON ----------
    out_dir = os.path.dirname(output_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Converted {len(data)} rows → {output_json_path}")


if __name__ == "__main__":
   jsonl_to_sorted_json(
        input_jsonl_path="claude4/vicuna_llama/lose_pairwise_results_ori_rbpo_classified.jsonl",
        output_json_path="claude4/anal_results/analysis.json"
    )




