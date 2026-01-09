import re
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from ranking_utils import read_jsonl

device = 'cuda:0'
model_name = "prometheus-eval/prometheus-7b-v2.0"

input_jsonl = "optimized_prompts_llama2_7b_res.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"

# ==== PROMPT TEMPLATE (Prometheus Pairwise) ====
PROMETHEUS_PROMPT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), a reference answer, and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A, Response B, and the Reference Answer. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_a}

###Response B:
{response_b}

###Reference Answer:
A helpful, accurate, and well-structured response that directly addresses the user's question.

###Score Rubric:
Which response better addresses the user's question in terms of helpfulness, relevance, accuracy, and clarity?

###Feedback: """

SYSTEM_MESSAGE = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

# ==== LOAD MODEL ====
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH)


def judge_pair(instruction: str, response_a: str, response_b: str) -> str:
    """
    So sánh 2 responses và trả về winner.
    Returns: "A", "B", hoặc None nếu không parse được
    """
    user_message = PROMETHEUS_PROMPT_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b
    )

    # Prometheus dùng Mistral format
    conversation = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message}
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,  # Prometheus sinh feedback dài hơn
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0
        )

    completion = tokenizer.decode(
        generation[0][len(input_ids[0]):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Parse kết quả [RESULT] A hoặc [RESULT] B
    match = re.search(r'\[RESULT\]\s*([AB])', completion)
    if match:
        return match.group(1)
    return None


def run_prometheus_ranking_loop(
    rows,
    output_jsonl,
    get_instruction_fn=lambda item: item["prompt"],
    get_prompt_1_fn=lambda item: item["prompt"],
    get_prompt_2_fn=lambda item: item["optimized_prompt"],
    get_output_1_fn=lambda item: item["res"],
    get_output_2_fn=lambda item: item["optimized_res"],
    label_0="original_win",
    label_1="optimized_win",
    label_2="draw",
    save_winner_0=True,
    save_winner_1=False,
):
    """
    Chạy ranking loop sử dụng Prometheus model.
    So sánh 2 responses bằng LLM judge để xác định winner.
    Nếu 2 prompt giống nhau thì cho kết quả hòa.

    Args:
        rows: list các item cần ranking
        output_jsonl: file output
        get_instruction_fn: hàm lấy instruction (câu hỏi gốc) từ item
        get_prompt_1_fn: hàm lấy prompt_1 (original) từ item
        get_prompt_2_fn: hàm lấy prompt_2 (optimized) từ item
        get_output_1_fn: hàm lấy output_1 từ item
        get_output_2_fn: hàm lấy output_2 từ item
        label_0, label_1, label_2: tên labels cho summary
        save_winner_0, save_winner_1: quyết định lưu item khi thắng
    """
    total_0 = 0
    total_1 = 0
    total_2 = 0
    total_error = 0

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        pbar = tqdm(rows, desc="Ranking with Prometheus")

        for item in pbar:
            instruction = get_instruction_fn(item)
            prompt_1 = get_prompt_1_fn(item)
            prompt_2 = get_prompt_2_fn(item)
            output_1 = get_output_1_fn(item)
            output_2 = get_output_2_fn(item)

            # Nếu 2 prompt giống nhau thì cho kết quả hòa
            if prompt_1 == prompt_2:
                winner = 2
            else:
                # So sánh 2 lần với thứ tự đổi chỗ để xử lý position bias
                # Lần 1: A = original, B = optimized
                result_1 = judge_pair(instruction, output_1, output_2)
                # Lần 2: A = optimized, B = original (đổi chỗ)
                result_2 = judge_pair(instruction, output_2, output_1)

                # Kết hợp kết quả:
                # - result_1: A thắng = original thắng, B thắng = optimized thắng
                # - result_2: A thắng = optimized thắng, B thắng = original thắng
                vote_original = 0
                vote_optimized = 0

                if result_1 == "A":
                    vote_original += 1
                elif result_1 == "B":
                    vote_optimized += 1

                if result_2 == "A":
                    vote_optimized += 1
                elif result_2 == "B":
                    vote_original += 1

                # Xác định winner dựa trên votes
                if vote_original > vote_optimized:
                    winner = 0  # original thắng
                elif vote_optimized > vote_original:
                    winner = 1  # optimized thắng
                elif vote_original == vote_optimized and vote_original > 0:
                    winner = 2  # Hòa (cả 2 lần cho kết quả ngược nhau)
                else:
                    winner = None
                    total_error += 1

            # Cập nhật counters và lưu file
            item["winner"] = winner

            if winner == 0:
                total_0 += 1
                if save_winner_0:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif winner == 1:
                total_1 += 1
                if save_winner_1:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif winner == 2:
                total_2 += 1

            # Cập nhật tqdm
            pbar.set_postfix({
                label_0: total_0,
                label_1: total_1,
                label_2: total_2,
                "err": total_error
            })

        # ==== WRITE SUMMARY ====
        total_all = total_0 + total_1 + total_2

        summary = {
            label_0: total_0,
            label_1: total_1,
            label_2: total_2,
            "total": total_all,
            "parse_errors": total_error
        }
        f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

        if total_all > 0:
            summary_stats = {
                f"{label_0}_percent": total_0 / total_all * 100,
                f"{label_1}_percent": total_1 / total_all * 100,
                f"{label_2}_percent": total_2 / total_all * 100,
            }
            f_out.write(json.dumps(summary_stats, ensure_ascii=False) + "\n")

    print(f"\nDONE! Saved to: {output_jsonl}")
    print(f"{label_0}: {total_0} ({total_0/total_all*100:.1f}%)")
    print(f"{label_1}: {total_1} ({total_1/total_all*100:.1f}%)")
    print(f"{label_2}: {total_2} ({total_2/total_all*100:.1f}%)")
    print(f"Total: {total_all}")
    if total_error > 0:
        print(f"Parse errors: {total_error}")

    return total_0, total_1, total_2


# ==== RUN ====
if __name__ == "__main__":
    rows = read_jsonl(input_jsonl)

    run_prometheus_ranking_loop(
        rows=rows,
        output_jsonl=output_jsonl,
        get_instruction_fn=lambda item: item["prompt"],
        get_prompt_1_fn=lambda item: item["bpo_prompt"],
        get_output_1_fn=lambda item: item["bpo_res"],
        get_prompt_2_fn=lambda item: item["optimized_prompt"],
        get_output_2_fn=lambda item: item["optimized_res"],
        label_0="original_win",
        label_1="optimized_win",
        label_2="draw",
        save_winner_0=True,
        save_winner_1=False,
    )
