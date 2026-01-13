"""
Grid search cho M (số lượng paraphrase samples).
Chạy semantic clustering với M từ 2 đến 20, threshold=0.05, imp_enc=0.5.
"""
from semantic import step3_sbert_clustering
from ranking_infer_from_file import run_ranking_from_file

# ==== THAM SỐ ====
M_values = list(range(1, 21))  # M từ 2 đến 20
distance_threshold = 0.05
imp_enc = 0.5

# ==== LƯU KẾT QUẢ ====
results = []

for M in M_values:
    print(f"\n{'='*60}")
    print(f"Running: M={M}, threshold={distance_threshold}, imp_enc={imp_enc}")
    print(f"{'='*60}")

    # Chạy step3 với M
    step3_sbert_clustering(distance_threshold=distance_threshold, imp_enc=imp_enc, M=M)

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
        "M": M,
        "win": win_pct,
        "draw": draw_pct,
        "lose": lose_pct
    })

# ==== IN KẾT QUẢ ====
print(f"\n{'='*80}")
print("SEARCH M RESULTS")
print(f"{'='*80}")
print(f"{'M':<10}{'win':<15}{'draw':<15}{'lose':<15}")
print("-" * 55)

for r in results:
    print(f"{r['M']:<10}{r['win']:<15.2f}{r['draw']:<15.2f}{r['lose']:<15.2f}")

# Tìm best M (win cao nhất, lose thấp nhất)
best = max(results, key=lambda x: (x['win'], -x['lose']))
print(f"\nBest M: {best['M']}")
print(f"  win={best['win']:.2f}, draw={best['draw']:.2f}, lose={best['lose']:.2f}")
