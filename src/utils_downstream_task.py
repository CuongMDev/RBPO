import os
import json
import random
import time
import requests
from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice, prompt_template_bbh_yes_no,prompt_template_bbh_true_false, prompt_template_bbh_valid
from helper import DEEPSEEK, GSM8K, BBH, ARC_C, ARC_E, PIQA
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SORTING = "sorting"
PIQA = "piqa"
MATH = "math"
YES_NO = "yes_no"
TRUE_FALSE = "true_false"
VALID = "valid"
MULTIPLE_CHOICE = "multiple_choice"

template_type = {
    SORTING: ["word_sorting"], # NOTE: No template for this subtask
    PIQA: [PIQA], # NOTE: piqa template
    VALID: ['formal_fallacies'], # NOTE: select between "valid" and "invalid"
    TRUE_FALSE: ["boolean_expressions"], # NOTE: select between "True" and "False"
    MATH: [
        GSM8K,
        "multistep_arithmetic_two", 
        "object_counting"
    ], # NOTE: math template
    YES_NO: [
        "causal_judgement",
        'navigate',
        'web_of_lies', 
        'sports_understanding'
    ], # NOTE: select between "Yes" and "No"
    MULTIPLE_CHOICE: [
        ARC_C,
        ARC_E,
        'snarks',
        'ruin_names', 
        'hyperbaton', 
        'movie_recommendation',
        'penguins_in_a_table',
        'temporal_sequences',       
        'date_understanding',
        'disambiguation_qa', 
        'logical_deduction_three_objects',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'reasoning_about_colored_objects',
        'salient_translation_error_detection',
        'tracking_shuffled_objects_five_objects', 
        'tracking_shuffled_objects_seven_objects',
        'tracking_shuffled_objects_three_objects'
    ] # NOTE: select between Multiple choices, ARC
}

def check_main_task(task_name: str):
    for main_task, sub_tasks in template_type.items():
        if task_name in sub_tasks:
            return main_task
    return None

def format_prompt_template(
    task_name: str,
    item: dict,
    method_key: str,
):    
    assert item.get(f'{method_key}_prompt') is not None 
    prompt = item.get(f'{method_key}_prompt')
    
    task = check_main_task(task_name)
    if task == MATH:
        return prompt_template_gsm8k.format(Q=prompt)
    
    elif task == SORTING:
        return prompt
    
    elif task == YES_NO:
        return prompt_template_bbh_yes_no.format(Q=prompt)
    
    elif task == TRUE_FALSE:
        return prompt_template_bbh_true_false.format(Q=prompt)
    
    elif task == VALID:
        return prompt_template_bbh_valid.format(Q=prompt)

    elif task == PIQA:
        sol1 = item.get("sol1")
        sol2 = item.get("sol2")
        return prompt_template_piqa.format(Q=prompt, sol1=sol1, sol2=sol2)    
    
    elif task == MULTIPLE_CHOICE:
        def build_options_block(labels, texts):
            return "\n".join([f"##{l}: {t}" for l, t in zip(labels, texts)])
        def build_label_instruction(labels):
            return " or ".join([f"({l})" for l in labels])
        
        choices = item.get("choices")
        labels = choices.get("label")
        texts = choices.get("text")
        options_block = build_options_block(labels, texts)
        label_instruction = build_label_instruction(labels)
        return prompt_template_multiple_choice.format(Q=prompt,
            options_block=options_block,
            label_instruction=label_instruction
        )
        
    return prompt

def extract_brackets(obj: str):
    import re
    pattern = r'\((.*?)\)'
    match = re.search(pattern, obj)
    if match:
        return match.group(1).strip()
    else:
        return None

def format_piqa_response(response: str, item: dict):
    sol1 = item.get("sol1").lower()
    sol2 = item.get("sol2").lower()
    response = extract_brackets(response).lower()
    if response == "a" or response == "(a)":
        return sol1
    elif response == "b" or response == "(b)":
        return sol2

# def verify_piqa(item:dict):


downstream_verify_system_prompt = """
You are a helpful and precise assistant for verifying the correctness of the answer. Please strictly follow the format requirements to provide your answer.
"""
general_verify_prompt = """
Question: {question}
Expected Answer: {expected_answer}
Predicted Answer: {predicted_answer}

Task:
Determine whether the predicted answer correctly answers the question.

Important Instructions:
- The predicted answer may include extra, irrelevant, or hallucinated content.
- Extract ONLY the FIRST valid answer corresponding to the original question.
- If the answer is presented as a list (e.g., "1. Yes", "A. No"), use ONLY the first item.
- For numeric problems, extract the first occurring number.
- Ignore any content after the first answer, especially if patterns like "Q:", "A:", "####", or additional QA examples appear.
- If multiple answers exist, ONLY use the earliest valid one.
- Evaluate correctness based solely on this extracted answer.

Output format (STRICT):
Return a JSON list with exactly one object:
[
  {{
    "response": "Right" or "Wrong",
    "Explanation": "<brief explanation, <= 32 tokens>"
  }}
]

Rules:
- "response" must be exactly "Right" or "Wrong"
- Explanation must be concise (<= 32 tokens)
- Do NOT output anything outside the JSON list
"""

math_verify_prompt = """
Question: {question}
Expected Answer: {expected_answer}
Predicted Answer: {predicted_answer}

Task:
Determine whether the predicted answer is numerically correct.

Important Instructions:
- The predicted answer may contain extra or irrelevant content.
- ONLY consider the FIRST answer before any new question (e.g., "Q:").
- If multiple answers exist, ONLY use the earliest valid one.
- Otherwise, extract the FIRST number appearing in the answer.

Normalization Rules:
- Ignore units, currency symbols ($), commas.
- Convert both answers to numbers before comparison.
- Treat numerically equivalent values as correct (e.g., 18 == 18.0).

Evaluation:
- Compare the numeric value of predicted vs expected.

Output format (STRICT):
Return a JSON list with exactly one object:
[
  {{
    "response": "Right" or "Wrong",
    "Explanation": "<brief explanation, <= 32 tokens>"
  }}
]

Rules:
- Only "Right" or "Wrong"
- Explanation <= 32 tokens
- No extra text
"""

sorting_verify_prompt = """
Question: {question}
Expected Answer: {expected_answer}
Predicted Answer: {predicted_answer}

Task:
Determine whether the predicted answer correctly preserves the alphabetical order of all expected items.

Important Instructions:
- The predicted answer may contain extra or irrelevant content.
- Identify and extract the MAIN list of words that represents the sorted output.
- Ignore unrelated text before or after the list.

Normalization Rules:
- Extract sequences from both answers.
- Normalize by:
  - Lowercasing
  - Removing punctuation
  - Splitting by spaces or commas

Evaluation:
- Let E = expected sequence, P = predicted sequence.
- The answer is correct if:
  1. ALL elements in E appear in P (no missing elements)
  2. The order of elements in E is preserved in P (subsequence match)
- Extra elements in P are allowed.

Output format (STRICT):
Return a JSON list with exactly one object:
[
  {{
    "response": "Right" or "Wrong",
    "Explanation": "<brief explanation, <= 32 tokens>"
  }}
]

Rules:
- Only "Right" or "Wrong"
- Explanation <= 32 tokens
- No extra text
"""

def keyword_verify(data_path,task_name,method_key,is_bbh):
    assert os.path.exists(data_path), f"Data path does not exist: {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:2] # NOTE: Only verify the first 5 samples for quick testing. Remove [:5] to verify the entire dataset.
    for item in data:
        if is_bbh:
            expected_ans = item.get("target").strip().lower()
        elif task_name == ARC_C or task_name == ARC_E or task_name == GSM8K:
            expected_ans = item.get("answer").strip().lower()
        elif task_name == PIQA:
            expected_ans = item.get("label")
        else:
            expected_ans = item.get("answer").strip().lower()  # Default fallback
        
        item['keyword_verification'] = False
        predicted_ans = item.get(f'{method_key}_response')
        if predicted_ans is int and predicted_ans == expected_ans:
            item['keyword_verification'] = True

        elif predicted_ans and predicted_ans.strip().lower() == expected_ans:
            item['keyword_verification'] = True
    
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
# EXTRACT JSON
def extract_json(raw):
    """Extract JSON from API response with better error handling"""
    import json
    import re
    
    if not raw or not raw.strip():
        raise ValueError("Empty response from API")
    
    raw = raw.strip()
    
    # Try to parse raw as-is first
    try: 
        json.loads(raw)
        return raw
    except Exception:
        pass
    
    # Strip thinking tags
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    
    # Try after removing thinking tags
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    
    # Remove markdown code blocks
    match = re.search(r'```(?:json)?\s*(.*?)```', cleaned, flags=re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        try:
            json.loads(extracted)
            return extracted
        except Exception:
            pass
    
    # Try to find JSON array [...]
    match = re.search(r'\[.*\]', cleaned, flags=re.DOTALL)
    if match:
        try:
            json.loads(match.group(0))
            return match.group(0)
        except Exception:
            pass
    
    # Try to find JSON object {...}
    # Use non-greedy matching with proper bracket counting
    for i in range(len(cleaned)):
        if cleaned[i] == '{':
            count = 0
            for j in range(i, len(cleaned)):
                if cleaned[j] == '{':
                    count += 1
                elif cleaned[j] == '}':
                    count -= 1
                    if count == 0:
                        candidate = cleaned[i:j+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            pass
                        break
    
    # If all else fails, raise error with diagnostics
    raise ValueError(f"Could not extract valid JSON. Raw response: {raw[:300]}")

def generate_verification_response(system_prompt, user_prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"} # use only for deepseek
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Status: {response.status_code} | Body: {response.text[:300]}")
        response.raise_for_status()
    
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def llm_verify(data_path, task_name, method_key, is_bbh):
    assert os.path.exists(data_path), f"Data path does not exist: {data_path}"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f) # NOTE: Only verify the first 5 samples for quick testing. Remove [:5] to verify the entire dataset.
        
    task = check_main_task(task_name)
    
    error_count = 0
    success_count = 0
    
    for idx, item in enumerate(data):
        question = item.get(f'{method_key}_input_llm')
        predicted_ans = item.get(f'{method_key}_response')
        expected_ans = None
        
        if item.get(f'{method_key}_response') == "":
            item[f'{method_key}_verification_response'] = {
                "response": "Wrong",
                "Explanation": "No answer generated"
            }
            continue
        
        if is_bbh:
            expected_ans = item.get("target").strip().lower()
        elif task_name == ARC_C or task_name == ARC_E:
            expected_ans = item.get("answer").strip().lower()
        elif task_name == PIQA:
            expected_ans = item.get("label")
        elif task_name == GSM8K:
            expected_ans = item.get("final_answer").strip().lower()
        else:
            expected_ans = item.get("answer").strip().lower()  # NOTE: xu ly trong BBH
        
        if task == SORTING:
            downstream_verify_prompt_template = sorting_verify_prompt
        elif task == MATH:
            downstream_verify_prompt_template = math_verify_prompt
            expected_ans = item.get(f"{method_key}_final_response")
            question = item.get('raw_question')
        else:
            downstream_verify_prompt_template = general_verify_prompt
            
        
        # NOTE: call DeepSeek api to verify correctness
        # print(f"Verifying item with question: {question}")
        verify_prompt = downstream_verify_prompt_template.format(
            question=question,
            predicted_answer=predicted_ans,
            expected_answer=expected_ans
        )
        
        item['template'] = verify_prompt
        
        # Try API call with retries
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                raw_response = generate_verification_response(downstream_verify_system_prompt, verify_prompt)
                json_str = extract_json(raw_response)
                
                parsed_json = json.loads(json_str)
                # Handle both array format [{}] and object format {}
                if isinstance(parsed_json, list):
                    item[f'{method_key}_verification_response'] = parsed_json[0] if parsed_json else {"response": "Error", "Explanation": "Empty response"}
                else:
                    item[f'{method_key}_verification_response'] = parsed_json
                
                success = True
                success_count += 1
                
            except json.JSONDecodeError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\n[ERROR at sample {idx}] JSON decode failed after {max_retries} retries")
                    print(f"  Question: {question[:100]}...")
                    print(f"  Raw response: {raw_response[:200]}")
                    print(f"  Extracted JSON: {json_str[:200]}")
                    item[f'{method_key}_verification_response'] = {"response": "Error", "Explanation": "JSON parsing failed"}
                    error_count += 1
                else:
                    print(f"[RETRY {retry_count}/{max_retries}] Sample {idx} - Waiting 2 seconds...")
                    time.sleep(2)  # Wait before retry
            
            except Exception as e:
                print(f"[ERROR at sample {idx}] Unexpected error: {str(e)[:100]}")
                item[f'{method_key}_verification_response'] = {"response": "Error", "Explanation": str(e)[:50]}
                error_count += 1
                break
        
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n=== Verification completed ===")
    print(f"Path: {data_path}")
    print(f"Success: {success_count} | Errors: {error_count} | Total: {len(data)}")
    print(f"Results saved.")