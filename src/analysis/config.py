import os
from openai import OpenAI
from dotenv import load_dotenv

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

MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"
MAX_TOKENS = 2048
TEMPERATURE = 0.0

#  ================== PROMPTS (ENGLISH) ==================

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