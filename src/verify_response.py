import json
import os
import requests

MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"
INPUT_FILE = "src/llama_vs_vicuna/Llama-2-7b-chat-hf/dolly_eval/deepseek-chat/lose_pairwise_results_bpo_rbpo.json"
OUTPUT_FILE = "src/verify_response/Llama-2-7b-chat-hf/dolly_eval/bpo_rbpo.jsonl"
PROMPT_FILE = "src/response_eval.txt"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EXPECTED_CRITERIA = [
    "Correctness", "Relevance", "Completeness", "Clarity_Coherence",
    "Usefulness_Helpfulness", "Style_Tone", "Conciseness", "Safety_Compliance"
]

# ================= LOAD API KEY =================
API_KEY_FILE = "src/key.txt"

with open(API_KEY_FILE, "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()

if not API_KEY:
    raise ValueError(f"API key rỗng. Kiểm tra lại file: {API_KEY_FILE}")

# ================= LOAD PROMPT FROM FILE =================
if not os.path.exists(PROMPT_FILE):
    raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

_prompt_vars = {}
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    exec(f.read(), _prompt_vars)

SYSTEM_PROMPT = _prompt_vars["SYSTEM_PROMPT"]

print("Loaded SYSTEM_PROMPT from response_eval.txt")
print(SYSTEM_PROMPT[:200], "\n---")

# ================= USER PROMPT =================
def build_user_prompt(item):
    # prompt = item.get("org_prompt", "")
    # response_A = item.get("res_0", "")
    # response_B = item.get("res_1", "")

    # if not prompt:
    #     raise ValueError(f"Missing org_prompt. Keys: {item.keys()}")

    return f"""
Prompt_A (used to generate Response_A):
\"\"\"{item.get("prompt_0", "")}\"\"\"

Response_A:
\"\"\"{item.get("res_0", "")}\"\"\"

Prompt_B (used to generate Response_B):
\"\"\"{item.get("prompt_1", "")}\"\"\"

Response_B:
\"\"\"{item.get("res_1", "")}\"\"\"

IMPORTANT RULES:
- Judge Response_A ONLY based on Prompt_A
- Judge Response_B ONLY based on Prompt_B
- Do NOT compare Response_A and Response_B
- Do NOT use any other information
- Use the SAME scoring scale for both

Return JSON ONLY in the following format:

{{
  "response_A": {{
    "Correctness": 0.0,
    "Relevance": 0.0,
    "Completeness": 0.0,
    "Clarity_Coherence": 0.0,
    "Usefulness_Helpfulness": 0.0,
    "Style_Tone": 0.0,
    "Conciseness": 0.0,
    "Safety_Compliance": 0.0
  }},
  "response_B": {{
    "Correctness": 0.0,
    "Relevance": 0.0,
    "Completeness": 0.0,
    "Clarity_Coherence": 0.0,
    "Usefulness_Helpfulness": 0.0,
    "Style_Tone": 0.0,
    "Conciseness": 0.0,
    "Safety_Compliance": 0.0
  }}
}}

STRICT:
- JSON only
- No markdown
- No explanation
- One decimal place
"""

# ================= GENERATION =================
def generate(system_prompt, user_prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Status: {response.status_code} | Body: {response.text[:300]}")
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# ================= EXTRACT JSON =================
import re

def extract_json(raw):
    """Strip thinking tags, markdown code blocks, lấy chỉ phần JSON"""
    # Bỏ <think>...</think>
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    # Bỏ ```json ... ``` hoặc ``` ... ```
    match = re.search(r'```(?:json)?\s*(.*?)```', raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Nếu không có code block, tìm JSON object đầu tiên
    match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    if match:
        return match.group(0).strip()
    return raw.strip()

def validate_schema(parsed):
    """Kiểm tra parsed JSON có đúng format không"""
    if not isinstance(parsed, dict):
        return False
    for key in ["response_A", "response_B"]:
        if key not in parsed:
            return False
        if not isinstance(parsed[key], dict):
            return False
        for criteria in EXPECTED_CRITERIA:
            if criteria not in parsed[key]:
                return False
            if not isinstance(parsed[key][criteria], (int, float)):
                return False
    return True

# Keyword mapping: model tự sinh criteria → map về criteria đúng
CRITERIA_MAP = {
    "Correctness":            ["correctness", "accuracy", "factual", "correct", "truthfulness", "honesty"],
    "Relevance":              ["relevance", "relevant", "on_topic", "topicality"],
    "Completeness":           ["completeness", "complete", "coverage", "thoroughness", "recall"],
    "Clarity_Coherence":      ["clarity", "coherence", "clarity_coherence", "readability", "structure", "clear"],
    "Usefulness_Helpfulness": ["usefulness", "helpfulness", "useful", "helpful", "instruction_following", "practicality"],
    "Style_Tone":             ["style", "tone", "style_tone", "formality", "politeness"],
    "Conciseness":            ["conciseness", "concise", "brevity", "verbosity"],
    "Safety_Compliance":      ["safety", "safety_compliance", "compliance", "bias", "harmful", "safe"]
}

def map_to_schema(raw_scores):
    """Map criteria của model về schema đúng"""
    mapped = {c: None for c in EXPECTED_CRITERIA}

    for model_key, model_val in raw_scores.items():
        if not isinstance(model_val, (int, float)):
            continue
        model_key_lower = model_key.lower()

        # Tìm criteria phù hợp nhất
        for expected, keywords in CRITERIA_MAP.items():
            if mapped[expected] is not None:
                continue  # đã assign rồi, skip
            if model_key_lower in keywords or model_key_lower == expected.lower():
                mapped[expected] = float(model_val)
                break

    # Các criteria chưa map được → dùng average của các scores đã map
    mapped_values = [v for v in mapped.values() if v is not None]
    avg = sum(mapped_values) / len(mapped_values) if mapped_values else 0.0

    for c in EXPECTED_CRITERIA:
        if mapped[c] is None:
            mapped[c] = round(avg, 1)

    return mapped

# ================= FALLBACK =================
def empty_schema():
    return {
        "response_A": {c: 0.0 for c in EXPECTED_CRITERIA},
        "response_B": {c: 0.0 for c in EXPECTED_CRITERIA}
    }

# check đầy đủ schema không
def is_complete(candidate):
    if not isinstance(candidate, dict):
        return False
    for key in ["response_A", "response_B"]:
        if key not in candidate:
            return False
        for c in EXPECTED_CRITERIA:
            if c not in candidate[key]:
                return False
            if not isinstance(candidate[key][c], (int, float)):
                return False
    return True
# ================= LOAD PROCESSED =================
def load_processed_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj.get("id"))
            except:
                continue
    return ids

# ================= DECIDE WINNER =================
def decide_winner_from_scores(llm_eval, threshold=0.01):
    """
    So sánh scores → return winner
    0 = draw
    1 = response_A win (res_0)
    2 = response_B win (res_1)
    """
    score_A = sum(llm_eval["response_A"].values()) / len(llm_eval["response_A"])
    score_B = sum(llm_eval["response_B"].values()) / len(llm_eval["response_B"])

    diff = score_A - score_B
    if abs(diff) < threshold:
        return 2  # draw
    elif diff > 0:
        return 0  # response_A win
    else:
        return 1  # response_B win

# ================= MAIN =================
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_ids = load_processed_ids(OUTPUT_FILE)
    print(f"Already processed: {len(processed_ids)} records")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, item in enumerate(data):
            item_id = item.get("id", i + 1)

            if item_id in processed_ids:
                continue

            print(f"Processing {i+1}/{len(data)} | ID={item_id}")

            parsed = None
            candidate = None

            for attempt in range(2):  # retry tối đa 1 lần
                raw = generate(SYSTEM_PROMPT, build_user_prompt(item))
                print(f"  raw (attempt {attempt+1}): {raw[:200]}")

                try:
                    candidate = json.loads(extract_json(raw))

                    if is_complete(candidate):
                        parsed = candidate
                        break

                    # nếu có đủ A và B nhưng thiếu / sai criteria → map
                    if isinstance(candidate, dict) and "response_A" in candidate and "response_B" in candidate:
                        mapped = {
                            "response_A": map_to_schema(candidate["response_A"]),
                            "response_B": map_to_schema(candidate["response_B"])
                        }
                        if is_complete(mapped):
                            parsed = mapped
                            print("  ✓ Mapped schema")
                            break

                except Exception as e:
                    print(f"  ⚠ Parse fail (attempt {attempt+1}): {e}")

            # fallback cuối cùng
            if parsed is None:
                print(f"  ❌ ID={item_id}: dùng fallback")
                parsed = empty_schema()

            result = {
                "id": item_id,
                "org_prompt": item.get("org_prompt"),
                "prompt_0": item.get("prompt_0"),
                "res_0": item.get("res_0"),
                "prompt_1": item.get("prompt_1"),
                "res_1": item.get("res_1"),
                "winner_before": item.get("winner"),
                "winner": decide_winner_from_scores(parsed),  # 0/1/2
                "llm_evaluation": parsed
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

if __name__ == "__main__":
    main()