import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Please set it via environment variable."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MODEL_NAME = "xiaomi/mimo-v2-flash:free"
MAX_TOKENS = 2048
TEMPERATURE = 0.0

# ================== PROMPTS (ENGLISH) ==================

SYSTEM_PROMPT = (
"You are a prompt analysis expert. "
"Your task is to classify how prompt_1 has been modified "
"compared to prompt_0. "
"You may perform internal reasoning but MUST NOT reveal your chain of thought. "
"ONLY return valid JSONL. "
"Classify each prompt pair in the list according to the SYSTEM_PROMPT and return JSONL only for each pair."
)

USER_PROMPT_TEMPLATE = """
prompt_0 (original prompt):
\"\"\"{prompt_0}\"\"\"

prompt_1 (modified prompt):
\"\"\"{prompt_1}\"\"\"

Task:
Classify the types of edits that have been applied in prompt_1 compared to prompt_0.

================= EDIT CATEGORIES (USE ONLY THE LABELS BELOW) =================

1.Intensification (Increased requirement strength):
The prompt makes the original task stronger or more demanding (e.g., “more detailed,” “deeper,” “more thorough”) without changing what is being asked.

2.Verb_Substitution (Core verb replacement):
The main verb of the prompt is replaced with a near-synonym while preserving the same intent and scope.

3.Aspect_Expansion (Content aspect expansion):
The prompt adds specific aspects, dimensions, or topics that were not present in the original prompt (e.g., listing elements to be analyzed).

4.Depth_Requirement (Deeper analysis requirement):
The prompt asks for a more detailed or in-depth explanation WITHOUT adding new content aspects.

5.Output_Structuring (Output structure enforcement):
The prompt requires the answer to follow a specific structure (e.g., step-by-step, bullet points, with illustrative examples).

6.Secondary_Objective (Secondary quality objective):
The prompt adds how the answer should be written, such as its style, tone, clarity, or suitability for a reader, not what content to include.

7.Instructional_Framing (Command-style framing):
The prompt’s main clause is shifted from a question to a direct, task-oriented command, rather than simply adding follow-up constraints (e.g., “Write…”, “Present…”, “Generate…”, “Provide…”).

8.Implication_Expansion (Implication/impact expansion):
The prompt asks to analyze consequences, social or ethical impacts, or long-term effects of the issue.

9.Audience_Specification (Audience specification):
The prompt adds information about the target audience to adjust the level and style of the response (e.g., “for beginners”, “for high school students”, “for non-technical readers”, ...).

10.Example_Request (Request for examples):
The prompt adds a requirement to provide concrete examples to illustrate the content (e.g., “with examples”, “give a concrete example”, ...).

11.Scope_Narrowing (Scope narrowing):
The prompt limits the topic or context to narrow the space of the answer (e.g., “in the context of healthcare only”, “focusing only on recent studies”, ...).

12.Minimal_Change (Near-identical change):
The prompt remains almost the same, with only very minor edits (e.g., punctuation, connectors, or changes that do not affect meaning).

13.Unclear_or_Other (Unclear / other):
The modification is unclear, ambiguous, or does not fit any of the categories above.

================= OUTPUT FORMAT (JSONL ONLY) =================

{{
  "labels": ["<label1>", "<label2>"],
  "descriptions": {{
"<label>": "<exact only one description in Vietnamese>"
  }},
  "evidence": {{
    "<label>": "<short quote from prompt_1 (max 12 words)>"
  }}
}}

================= RULES =================
-Use only the labels listed above.
-Each description must be exactly one sentence, in Vietnamese.
-Evidence must be quoted verbatim from prompt_1, ONLY the minimal changed phrase.
- Do not quote the full sentence.
-If there is almost no change, use only ["Minimal_Change"].
-"Minimal_Change" and "Unclear_or_Other" MUST NOT be combined with other labels.
"""

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

    classification = classify_pair(prompt_0, prompt_1)
    parsed = json.loads(classification)

    return {
        "id": item.get("id"),
        "prompt_0": prompt_0,
        "prompt_1": prompt_1,
        "classification": parsed
    }


def batch_data_for_analysis(data, batch_size):
    """
    Chia data thành các batch nhỏ để phân tích.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
        
        
# ================== MAIN PIPELINE ==================
    
def classify_dataset(input_json_path: str,
                     output_jsonl_path: str,
                     batch_size: int,
                     max_workers: int = 4):
    """
    Đọc dataset prompt pair, phân loại song song và ghi ra JSONL.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for batch in batch_data_for_analysis(data, batch_size):
        print(f"Processing batch size = {len(batch)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one_item, item) for item in batch]

            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                    print(f"[OK] id={r['id']}")

                except Exception as e:
                    print(f"[ERROR] {e}")

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved results to: {output_jsonl_path}")

# ================== ENTRY POINT ==================

if __name__ == "__main__":
    
    folder_paths = [
        "claude4/vicuna_llama_claude4",
        "claude4/vicuna_vicuna_claude4",
        "claude4/dolly_vicuna_claude4",
        "claude4/dolly_llama_claude4"]
    
    input = "lose_pairwise_results_ori_rbpo_preprocessed.json"
    output = "lose_pairwise_results_ori_rbpo_classified.jsonl"
    
    for folder in folder_paths:
        INPUT_PATH = folder+input
        OUTPUT_PATH = folder+output
        classify_dataset(
            input_json_path=INPUT_PATH,
            output_jsonl_path=OUTPUT_PATH,
            batch_size=2,     # mỗi batch 1 item
            max_workers=2     # 4 request song song
        )

