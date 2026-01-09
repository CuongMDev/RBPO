"""
Grid search cho distance_threshold và imp_enc.
Chạy semantic clustering với từng cặp tham số, sau đó đánh giá bằng ranking_infer_from_file.
"""
from semantic import step3_sbert_clustering
from ranking_infer_from_file import run_ranking_from_file

# ==== THAM SỐ CẦN THỬ ====
distance_thresholds = [0.03, 0.05, 0.07, 0.1]
imp_encs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0]

# ==== LƯU KẾT QUẢ ====
results = []

for dist_thresh in distance_thresholds:
    for imp_enc in imp_encs:
        print(f"\n{'='*60}")
        print(f"Running: distance_threshold={dist_thresh}, imp_enc={imp_enc}")
        print(f"{'='*60}")

        # Chạy step3 trực tiếp
        step3_sbert_clustering(distance_threshold=dist_thresh, imp_enc=imp_enc)

        # Chạy ranking_infer_from_file và lấy kết quả
        total_0, total_1, total_2 = run_ranking_from_file()

        total_all = total_0 + total_1 + total_2
        if total_all > 0:
            win_pct = total_1 / total_all * 100
            draw_pct = total_2 / total_all * 100
            lose_pct = total_0 / total_all * 100
        else:
            win_pct = draw_pct = lose_pct = 0

        results.append({
            "distance_threshold": dist_thresh,
            "imp_enc": imp_enc,
            "win": win_pct,
            "draw": draw_pct,
            "lose": lose_pct
        })

# ==== IN KẾT QUẢ ====
print(f"\n{'='*80}")
print("GRID SEARCH RESULTS")
print(f"{'='*80}")
print(f"{'dist_thresh':<15}{'imp_enc':<10}{'win':<10}{'draw':<10}{'lose':<10}")
print("-" * 55)

for r in results:
    print(f"{r['distance_threshold']:<15}{r['imp_enc']:<10}{r['win']:<10.2f}{r['draw']:<10.2f}{r['lose']:<10.2f}")

# Tìm best config (win cao nhất, lose thấp nhất)
best = max(results, key=lambda x: (x['win'], -x['lose']))
print(f"\nBest config: distance_threshold={best['distance_threshold']}, imp_enc={best['imp_enc']}")
print(f"  win={best['win']:.2f}, draw={best['draw']:.2f}, lose={best['lose']:.2f}")
