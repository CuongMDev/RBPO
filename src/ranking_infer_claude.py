import torch
import json
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from ranking_utils import (
    load_ranking_prompt,
    read_jsonl,
    run_ranking_loop,
    run_followup_inference
)

ANTHROPIC_API_KEY = "API_KEY"
client = OpenAI(
    api_key=ANTHROPIC_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

device = 'cuda:0'
model_name = "Qwen/Qwen3-4B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH
)

def run_with_claude(prompt):
    """
    Gọi Claude để lấy reasoning, sau đó dùng local model để extract boxed answer
    """

    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4",
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    reasoning = response.choices[0].message.content
    return run_followup_inference(
        reasoning,
        model,
        tokenizer,
        device
    )

# ==== LOAD PROMPT TEMPLATE ====
raw_prompt = load_ranking_prompt()

input_jsonl = "responses_with_semantic.jsonl"

output_jsonl = "lose_pairwise_results.jsonl"

output_result= "results.json"

if __name__ == "__main__":
    rows = read_jsonl(input_jsonl)

    stats = {}

    # ===== 0. ORIGINAL vs BPO =====
    orig_0, bpo_0, draw_0 = run_ranking_loop(
        rows=rows,
        eval_fn=run_with_claude,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["prompt"],
        get_output_1_fn=lambda item: item["response_original"],
        get_prompt_2_fn=lambda item: item["bpo_prompt"],
        get_output_2_fn=lambda item: item["bpo_response"],
        label_0="original_win",
        label_1="bpo_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
        handle_bias=False
    )

    stats["original_vs_bpo"] = {
        "original_win": orig_0,
        "bpo_win": bpo_0,
        "draw": draw_0,
    }
    
    # ===== 1. ORIGINAL vs RBPO =====
    bpo_1, rbpo_1, draw_1 = run_ranking_loop(
        rows=rows,
        eval_fn=run_with_claude,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["prompt"],
        get_output_1_fn=lambda item: item["response_original"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_2_fn=lambda item: item["optimized_response"],
        label_0="original_win",
        label_1="rbpo_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
        handle_bias=False
    )

    stats["original_vs_rbpo"] = {
        "bpo_win": bpo_1,
        "rbpo_win": rbpo_1,
        "draw": draw_1,
    }

    # ===== 2. BPO vs RBPO =====
    bpo_2, rbpo_2, draw_2 = run_ranking_loop(
        rows=rows,
        eval_fn=run_with_claude,
        raw_prompt=raw_prompt,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["bpo_prompt"],
        get_output_1_fn=lambda item: item["bpo_response"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_2_fn=lambda item: item["optimized_response"],
        label_0="bpo_win",
        label_1="rbpo_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
        handle_bias=False
    )

    stats["bpo_vs_rbpo"] = {
        "bpo_win": bpo_2,
        "rbpo_win": rbpo_2,
        "draw": draw_2,
    }


    # ===== SAVE RESULT =====
    with open(output_result, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved ranking results to {output_result}")