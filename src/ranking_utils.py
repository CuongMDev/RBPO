import re
import json
from tqdm import tqdm
from utils import make_prompt_template

# ==== ĐỌC PROMPT TEMPLATE TỪ FILE ====
def load_ranking_prompt(prompt_file="ranking_prompt.txt"):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

# ==== FILL PROMPT TEMPLATE ====
def fill_prompt(raw_prompt, instruction, output_1, output_2):
    """Điền instruction và 2 outputs vào template"""
    prompt = raw_prompt
    prompt = prompt.replace('{instruction}', instruction)
    prompt = prompt.replace('{output_1}', output_1)
    prompt = prompt.replace('{output_2}', output_2)
    return prompt

# ==== EXTRACT WINNER ====
def extract_winner(text):
    """
    Tìm \\boxed{…} đầu tiên trong text, xem có số 1 hay 2 bên trong.
    Trả về:
        0 nếu có số 1 (model_1 thắng)
        1 nếu có số 2 (model_2 thắng)
        2 nếu có 'both' (hòa)
        None nếu không tìm thấy
    """
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        content = m.group(1)
        if 'both' in content:
            return 2
        if '1' in content:
            return 0
        elif '2' in content:
            return 1
    return None

# ==== READ JSONL ====
def read_jsonl(filepath):
    """Đọc file JSONL và trả về list"""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# ==== RANK PAIR WITH POSITION BIAS HANDLING ====
def rank_pair(eval_fn, raw_prompt, instruction, output_1, output_2, handle_bias=True):
    """
    So sánh 2 outputs với xử lý position bias (đổi chỗ 2 lần).

    Args:
        eval_fn: hàm nhận prompt và trả về result_text
        raw_prompt: prompt template
        instruction: câu hỏi gốc
        output_1: output thứ nhất (original)
        output_2: output thứ hai (optimized)
        handle_bias: nếu True, chạy 2 lần (đổi chỗ) để xử lý position bias

    Returns:
        winner: 0 = output_1 thắng, 1 = output_2 thắng, 2 = hòa, None = lỗi
    """
    # Lần 1: model_1 = output_1, model_2 = output_2
    prompt_order1 = fill_prompt(raw_prompt, instruction, output_1, output_2)
    result_1 = eval_fn(prompt_order1)
    winner_1 = extract_winner(result_1)

    if not handle_bias:
        # Không xử lý bias, trả về kết quả lần 1
        return winner_1

    # Lần 2: model_1 = output_2, model_2 = output_1 (đổi chỗ)
    prompt_order2 = fill_prompt(raw_prompt, instruction, output_2, output_1)
    result_2 = eval_fn(prompt_order2)
    winner_2 = extract_winner(result_2)

    # Kết hợp kết quả:
    # - winner_1: 0 = output_1 thắng, 1 = output_2 thắng
    # - winner_2: 0 = output_2 thắng, 1 = output_1 thắng (vì đổi chỗ)
    vote_1 = 0
    vote_2 = 0

    if winner_1 == 0:
        vote_1 += 1
    elif winner_1 == 1:
        vote_2 += 1

    if winner_2 == 0:
        vote_2 += 1  # Đổi chỗ nên model_1 thắng = output_2 thắng
    elif winner_2 == 1:
        vote_1 += 1  # Đổi chỗ nên model_2 thắng = output_1 thắng

    # Xác định winner
    if vote_1 > vote_2:
        return 0  # output_1 thắng
    elif vote_2 > vote_1:
        return 1  # output_2 thắng
    elif vote_1 == vote_2 and (winner_1 is not None and winner_2 is not None):
        return 2  # Hòa
    else:
        return None  # Lỗi parse


# ==== RANKING LOOP ====
def run_ranking_loop(
    rows,
    eval_fn,
    raw_prompt,
    output_jsonl,
    get_instruction_fn=lambda item: item["prompt"],
    get_prompt_1_fn=lambda item: item["prompt"],
    get_prompt_2_fn=lambda item: item["optimized_prompt"],
    get_output_1_fn=lambda item: item["res"],
    get_output_2_fn=lambda item: item["optimized_res"],
    label_0="original_win",
    label_1="optimized_win",
    label_2="draw",
    save_winner_0=True,  # Lưu item khi winner=0
    save_winner_1=False,  # Lưu item khi winner=1
    handle_bias=True,  # Xử lý position bias (chạy 2 lần đổi chỗ)
):
    """
    Chạy ranking loop chung cho tất cả các file ranking.

    Args:
        rows: list các item cần ranking
        eval_fn: hàm nhận prompt và trả về result_text (để extract winner)
        raw_prompt: prompt template đã load
        output_jsonl: file output
        get_instruction_fn: hàm lấy instruction từ item
        get_output_1_fn: hàm lấy output_1 từ item
        get_output_2_fn: hàm lấy output_2 từ item
        label_0, label_1, label_2: tên labels cho summary
        save_winner_0, save_winner_1: quyết định lưu item khi thắng
        handle_bias: nếu True, chạy 2 lần (đổi chỗ) để xử lý position bias
    """
    total_0 = 0
    total_1 = 0
    total_2 = 0

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        pbar = tqdm(rows, desc="Ranking pairs")

        for item in pbar:
            instruction = get_instruction_fn(item)
            prompt_1 = get_prompt_1_fn(item)
            prompt_2 = get_prompt_2_fn(item)
            output_1 = get_output_1_fn(item)
            output_2 = get_output_2_fn(item)
            if prompt_1 == prompt_2 or output_1 == output_2:
                winner = 2
            else:
                winner = rank_pair(eval_fn, raw_prompt, instruction, output_1, output_2, handle_bias=handle_bias)

            # Cập nhật counters và lưu file
            save_item = {
                "org_prompt": instruction,
                "prompt_0": prompt_1,
                "res_0": output_1,
                "prompt_1": prompt_2,
                "res_1": output_2,
                "winner": winner
            }
            if winner == 0:
                total_0 += 1
                if save_winner_0:
                    f_out.write(json.dumps(save_item, ensure_ascii=False) + "\n")
            elif winner == 1:
                total_1 += 1
                if save_winner_1:
                    f_out.write(json.dumps(save_item, ensure_ascii=False) + "\n")
            elif winner == 2:
                total_2 += 1
            else:
                print(f"[WARNING] Cannot extract winner for: {instruction[:100]}...")

            f_out.write(json.dumps(save_item, ensure_ascii=False) + "\n")

            # Cập nhật tqdm
            pbar.set_postfix({
                label_0: total_0,
                label_1: total_1,
                label_2: total_2
            })

        # ==== WRITE SUMMARY ====
        total_all = total_0 + total_1 + total_2

        summary = {
            label_0: total_0,
            label_1: total_1,
            label_2: total_2
        }

        if total_all > 0:
            summary_percent = {
                f"{label_0}_percent": total_0 / total_all * 100,
                f"{label_1}_percent": total_1 / total_all * 100,
                f"{label_2}_percent": total_2 / total_all * 100
            }
            f_out.write(json.dumps(summary_percent, ensure_ascii=False) + "\n")

    print(f"DONE! Saved to: {output_jsonl}")
    print(f"{label_0}: {total_0}")
    print(f"{label_1}: {total_1}")
    print(f"{label_2}: {total_2}")
    print(f"Total: {total_all}")

    return total_0, total_1, total_2


# ==== FOLLOWUP INFERENCE (chung cho tất cả) ====
def run_followup_inference(reasoning_text, model, tokenizer, device='cuda:0'):
    """
    Thêm followup prompt và infer để extract boxed answer.
    Dùng chung cho cả local model và Gemini.
    """
    # Nối câu bắt buộc boxed
    followup_prompt = reasoning_text + "\nSo among 'model_1', 'model_2' and 'both', we should choose \\boxed{"

    # Infer tiếp
    model_inputs = tokenizer(followup_prompt, return_tensors="pt", truncation=True, max_length=5000).to(device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=50,
        do_sample=False
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Lấy phần model sinh thêm
    final_output = followup_prompt + decoded[len(followup_prompt):].strip()

    return final_output


# ==== FIRST INFERENCE (chỉ local models) ====
def run_first_inference(prompt, model, tokenizer, device='cuda:0'):
    """
    Infer lần 1: prompt -> reasoning text.
    Chỉ dùng cho local models (Gemma, Llama, etc.)
    """
    # Chuẩn bị prompt với chat template
    formatted_prompt = make_prompt_template(prompt, add_system_prompt=False, add_ranking_system_prompt=False)
    formatted_prompt = tokenizer.apply_chat_template(
        formatted_prompt, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize và infer
    model_inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=False).to(device)
    input_len = model_inputs["input_ids"].shape[1]
    output = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    # Tách phần mới sinh (bỏ prompt)
    generated_ids = output[0][input_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # print(tokenizer.decode(output[0], skip_special_tokens=True))

    return decoded


# ==== EVAL MODEL (kết hợp 2 bước) ====
def run_eval_model(prompt, model, tokenizer, device='cuda:0'):
    """
    Hàm eval chung cho các local models.
    Kết hợp: first inference -> followup inference
    """
    reasoning = run_first_inference(prompt, model, tokenizer, device)
    return run_followup_inference(reasoning, model, tokenizer, device)
