"""
Ranking inference từ file all_ranking_results.jsonl (đã có sẵn kết quả từ full_ranking.py)
Không cần chạy model nữa.
"""
import json
from tqdm import tqdm

# ==== FILES ====
input_semantic = "responses_with_semantic.jsonl"
input_rankings = "all_ranking_results.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"

# ==== LOAD DATA ====
def load_data():
    """Load semantic data và ranking results"""
    # Load semantic data
    semantic_rows = []
    with open(input_semantic, "r", encoding="utf-8") as f:
        for line in f:
            semantic_rows.append(json.loads(line))

    # Load ranking results (mỗi dòng là array winners cho 1 item)
    ranking_rows = []
    with open(input_rankings, "r", encoding="utf-8") as f:
        for line in f:
            ranking_rows.append(json.loads(line))

    return semantic_rows, ranking_rows


def run_ranking_from_file():
    """
    Chạy ranking bằng cách đọc kết quả từ file đã có sẵn.

    Logic:
    - all_ranking_results.jsonl chứa winners cho từng paraphrase so với original
    - winners[i]: 0 = paraphrase thắng, 1 = original thắng, 2 = hòa
    - Cần tìm winner cho optimized_prompt vs original
    """
    semantic_rows, ranking_rows = load_data()

    if len(semantic_rows) != len(ranking_rows):
        print(f"[WARNING] Số dòng không khớp: semantic={len(semantic_rows)}, ranking={len(ranking_rows)}")

    total_0 = 0  # original win
    total_1 = 0  # optimized win
    total_2 = 0  # draw

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, (item, winners) in enumerate(tqdm(
            zip(semantic_rows, ranking_rows),
            total=min(len(semantic_rows), len(ranking_rows)),
            desc="Processing rankings"
        )):
            prompt = item["prompt"]
            response_original = item["response_original"]
            optimized_prompt = item["optimized_prompt"]
            optimized_response = item["optimized_response"]
            paraphrase_prompts = item["paraphrase_prompts"]

            # Tìm index của optimized_prompt trong paraphrase_prompts
            optimized_idx = None
            for i, p in enumerate(paraphrase_prompts):
                if p.strip() == optimized_prompt.strip():
                    optimized_idx = i
                    break

            if optimized_idx is None:
                print(f"[WARNING] Row {idx}: optimized_prompt không tìm thấy trong paraphrase_prompts")
                continue

            # Lấy winner từ ranking results
            # winners[i]: 0 = paraphrase thắng, 1 = original thắng, 2 = hòa/giống nhau
            winner_raw = winners[optimized_idx]

            # Chuyển đổi:
            # - winner_raw=0 (paraphrase thắng) -> winner=1 (optimized win)
            # - winner_raw=1 (original thắng) -> winner=0 (original win)
            # - winner_raw=2 (hòa) -> winner=2 (draw)
            if winner_raw == 0:
                winner = 1  # optimized win
            elif winner_raw == 1:
                winner = 0  # original win
            else:
                winner = 2  # draw

            # Cập nhật counters
            if winner == 0:
                total_0 += 1
            elif winner == 1:
                total_1 += 1
            else:
                total_2 += 1

            # Lưu kết quả
            save_item = {
                "org_prompt": prompt,
                "prompt_0": prompt,
                "res_0": response_original,
                "prompt_1": optimized_prompt,
                "res_1": optimized_response,
                "winner": winner
            }
            fout.write(json.dumps(save_item, ensure_ascii=False) + "\n")

        # ==== WRITE SUMMARY ====
        total_all = total_0 + total_1 + total_2

        if total_all > 0:
            summary_percent = {
                "original_win_percent": total_0 / total_all * 100,
                "rbpo_win_percent": total_1 / total_all * 100,
                "draw_percent": total_2 / total_all * 100
            }
            fout.write(json.dumps(summary_percent, ensure_ascii=False) + "\n")

    # ==== PRINT SUMMARY ====
    total_all = total_0 + total_1 + total_2
    if total_all > 0:
        win_pct = total_1 / total_all * 100
        draw_pct = total_2 / total_all * 100
        lose_pct = total_0 / total_all * 100
        print(f"\nDONE! Saved to: {output_jsonl}")
        print(f"Total: {total_all}")
        print(f"\nwin\tdraw\tlose")
        print(f"{win_pct:.2f}\t{draw_pct:.2f}\t{lose_pct:.2f}")

    return total_0, total_1, total_2


if __name__ == "__main__":
    run_ranking_from_file()
