import json
import os

from helper import eval_folder_name, experiment_file_name, mepo_folder_name

verify_keys_method = {
        "rbpo_bpo": ['rbpo_prompt', 'rbpo_response', 'bpo_prompt', 'bpo_response'], # đánh giá RBPO vs BPO
        "rbpo_mepo": ['rbpo_prompt', 'rbpo_response', 'mepo_prompt', 'mepo_response'], # đánh giá RBPO vs MEPO
        "rmepo_mepo": ['rmepo_prompt', 'rmepo_response', 'mepo_prompt', 'mepo_response'] # đánh giá RMEPO vs MEPO
    }

with open(experiment_file_name, "r") as f:
    lines = f.readlines()

for key, keys in verify_keys_method.items():
    print(f"\n=== VERIFY KEY: {key} ===")
    print(f"Verify keys: {keys}")
    for line in lines:
        path = line.strip()
        print(f"Processing: {path}")

        if not os.path.exists(f'{mepo_folder_name}/verify/mismatch/{path}_mismatch.json'):
            print(f"File not found: {mepo_folder_name}/verify/mismatch/{path}_mismatch.json")
            continue
            
        with open(f'{mepo_folder_name}/verify/mismatch/{path}_mismatch.json', "r", encoding="utf-8") as f:
            data = json.load(f)
            
        res = []
        for i,item in enumerate(data['mismatches']):
            winner_per_runs = item['winners_per_run']
            # lay winner nhieu lan nhat
            winner = max(set(winner_per_runs), key=winner_per_runs.count)
            idx = item['id']
            res.append({
                "id": idx,
                "final_winner": winner
            })
        # thong ke winner
        winner_counts = {}
        for item in res:
            winner = item['final_winner']
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

        output_path = f'{mepo_folder_name}/verify/final/{path}_final.json'
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "winner_counts": winner_counts,
                "mismatches": res
            }, f, ensure_ascii=False, indent=2)
        