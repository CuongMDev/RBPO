import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, MODEL_NAME, MAX_TOKENS, TEMPERATURE, client



#

# ================== CORE FUNCTIONS ==================

def run_with_claude(system_prompt: str, user_prompt: str) -> str:
    """
    Gọi Claude để phân loại prompt modification.
    Trả về raw JSONL text.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def classify_pair(prompt_0: str, prompt_1: str) -> str:
    """
    Build USER_PROMPT và gọi Claude cho 1 cặp prompt.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        prompt_0=prompt_0,
        prompt_1=prompt_1
    )
    result = run_with_claude(SYSTEM_PROMPT, user_prompt)
    return result

def process_one_item(item):
    prompt_0 = item.get("prompt_0")
    prompt_1 = item.get("prompt_1")
    winner = item.get("winner")

    classification = classify_pair(prompt_0, prompt_1)
    parsed = json.loads(classification)

    return {
        "id": item.get("id"),
        "prompt_0": prompt_0,
        "prompt_1": prompt_1,
        "winner": winner,
        "classification": parsed
    }


def batch_data_for_analysis(data, batch_size):
    """
    Chia data thành các batch nhỏ để phân tích.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
        
        
# ================== MAIN PIPELINE ==================
    
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


def classify_dataset(input_json_path: str,
                     output_json_path: str,
                     batch_size: int = 2,
                     max_workers: int = 3):
    """
    Đọc dataset prompt pair, phân loại song song và ghi ra JSON (list object).
    Skip an toàn nếu input file không tồn tại hoặc JSON lỗi.
    """

    # ---------- 1. Check input path ----------
    if not os.path.exists(input_json_path):
        print(f"[SKIP] Input file not found: {input_json_path}")
        return

    # ---------- 2. Load JSON an toàn ----------
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[SKIP] Invalid JSON in {input_json_path}: {e}")
        return

    if not isinstance(data, list) or len(data) == 0:
        print(f"[SKIP] Empty or invalid data list: {input_json_path}")
        return

    results = []

    # ---------- 3. Batch processing ----------
    for batch in batch_data_for_analysis(data, batch_size):
        print(f"Processing batch size = {len(batch)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_one_item, item)
                for item in batch
                if item.get("prompt_0") and item.get("prompt_1")
            ]

            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    print(f"[OK] id={r['id']}")
                except Exception as e:
                    print(f"[ERROR] {e}")

    # ---------- 4. Sort results ----------
    try:
        results.sort(key=lambda x: x.get("id", float("inf")))
    except Exception as e:
        print(f"[WARN] Sort failed: {e}")

    # ---------- 5. Ghi output (JSON) ----------
    out_dir = os.path.dirname(output_json_path)
    if out_dir:  # tránh WinError 3 khi path rỗng
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved {len(results)} rows to: {output_json_path}")
