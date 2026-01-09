"""
Benchmark so sánh thời gian giữa:
1. BPO đơn giản (infer_bpo): sinh 1 prompt tối ưu, không xử lý gì
2. RBPO (infer_rbpo): sinh M prompts, clustering, chọn best prompt
"""

import json
import time
import torch
from tqdm import tqdm

from infer_res_rbpo import (
    load_models,
    infer_bpo,
    infer_rbpo,
    M,
    device
)

# ============ CONFIG ============
input_jsonl = "testset/dolly_eval.json"
output_jsonl = "benchmark_results.jsonl"
num_samples = None  # Số samples để benchmark (set None để chạy hết)


def benchmark():
    # Load models
    print("Loading models...")
    bpo_model, bpo_tokenizer, sbert = load_models(device)

    # Read input
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        # for line in f:
            # data.append(json.loads(line))
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    print(f"\nBenchmarking on {len(data)} samples...")
    print(f"BPO: sinh 1 prompt tối ưu")
    print(f"RBPO: sinh {M} prompts, clustering, chọn best\n")

    # Benchmark results
    bpo_times = []
    rbpo_times = []
    results = []

    for item in tqdm(data, desc="Benchmarking"):
        prompt = item.get("text", item.get("instruction", ""))

        # === BPO ===
        torch.cuda.synchronize()
        start_bpo = time.perf_counter()

        bpo_result = infer_bpo(
            prompt,
            bpo_model, bpo_tokenizer,
            device=device
        )

        torch.cuda.synchronize()
        end_bpo = time.perf_counter()
        bpo_time = end_bpo - start_bpo
        bpo_times.append(bpo_time)

        torch.cuda.empty_cache()

        # === RBPO ===
        torch.cuda.synchronize()
        start_rbpo = time.perf_counter()

        rbpo_result = infer_rbpo(
            prompt,
            bpo_model, bpo_tokenizer,
            sbert,
            device=device
        )

        torch.cuda.synchronize()
        end_rbpo = time.perf_counter()
        rbpo_time = end_rbpo - start_rbpo
        rbpo_times.append(rbpo_time)

        torch.cuda.empty_cache()

        # Save result
        results.append({
            "prompt": prompt,
            "bpo_time": bpo_time,
            "rbpo_time": rbpo_time,
            "speedup": rbpo_time / bpo_time if bpo_time > 0 else 0,
            "bpo_prompt": bpo_result["optimized_prompt"][:200] + "...",
            "rbpo_prompt": rbpo_result["optimized_prompt"][:200] + "...",
            "rbpo_num_clusters": rbpo_result["num_clusters"]
        })

    # === Summary ===
    avg_bpo = sum(bpo_times) / len(bpo_times)
    avg_rbpo = sum(rbpo_times) / len(rbpo_times)
    speedup = avg_rbpo / avg_bpo if avg_bpo > 0 else 0

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Samples: {len(data)}")
    print(f"\nBPO (simple):")
    print(f"  - Avg time: {avg_bpo:.2f}s")
    print(f"  - Total time: {sum(bpo_times):.2f}s")
    print(f"\nRBPO (with clustering):")
    print(f"  - Avg time: {avg_rbpo:.2f}s")
    print(f"  - Total time: {sum(rbpo_times):.2f}s")
    print(f"\nSpeedup (RBPO/BPO): {speedup:.2f}x slower")
    print(f"Time overhead: {avg_rbpo - avg_bpo:.2f}s per sample")
    print("=" * 50)

    # Save to file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        summary = {
            "type": "summary",
            "num_samples": len(data),
            "bpo_avg_time": avg_bpo,
            "rbpo_avg_time": avg_rbpo,
            "bpo_total_time": sum(bpo_times),
            "rbpo_total_time": sum(rbpo_times),
            "speedup": speedup,
            "time_overhead_per_sample": avg_rbpo - avg_bpo
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nResults saved to: {output_jsonl}")

    # Cleanup
    del bpo_model, sbert
    torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
