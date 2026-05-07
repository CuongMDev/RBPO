# ⚙️ Cấu hình
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_PATH = "./hf_cache/model"
OPT_PROMPT_MODEL_CACHE_PATH = "./hf_cache/opt_prompt_model"
DATA_CACHE_PATH = "./hf_cache/data"
INSTRUCTION_DATA_PATH = "data/"       # file jsonl như mô tả ở trên

prompt_template_optimize = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"
prompt_template_vicuna = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {} 
ASSISTANT: """

# gsm8k, bbh-math
prompt_template_gsm8k = """
You are an expert of math problem solver.
Solve the problem step by step.

After giving the final answer, STOP immediately.
Do not generate any new question.

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A:
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A:
In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty\'s grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5

Q: {Q}
A: ####"""
# Output only the final numerical answer after #### (no explanation)

prompt_template_piqa = """
Choose the correct answer.
Return ONLY A or B. No explanation.

After giving the final answer, STOP immediately.
Do not generate any new question.

'When boiling butter, when it\'s ready, you can’
Option:
A: 'Pour it onto a plate’
B: 'Pour it into a jar'
Answer: B

'To permanently attach metal legs to a chair, you can’
Option:
A: 'Weld the metal together to get it to stay firmly in place’
B: 'Nail the metal together to get it to stay firmly in place'
Answer: A

'how do you indent something?’
Option:
A: 'leave a space before starting the writing’
B: 'press the spacebar'
Answer: A

{Q}
Option:
A: {sol1}
B: {sol2}
Answer:"""

# arc, bbh-multiple choice
prompt_template_multiple_choice = """
Answer the following question. No explanation. 
Return ONLY the correct option label in the format: (<label>)
Examples: (A), (B), (C)

After giving the final answer, STOP immediately.
Do not generate any new question. 

Question: {Q}
Options: {options_block}
Reply me with the option of the answer like {label_instruction}
Answer: """

prompt_template_bbh_yes_no = """
Question:
{Q}
Options:
- Yes
- No
Reply me with the option of the answer like Yes or No.
After giving the final answer, STOP immediately.
Do not generate any new question.
Answer:"""

prompt_template_bbh_true_false = """
Question:
{Q}
Options:
- True
- False
Reply me with the option of the answer like True or False.
After giving the final answer, STOP immediately.
Do not generate any new question.
Answer:"""

prompt_template_bbh_valid = """
Question:
{Q}
Options:
- Valid
- Invalid
Reply me with the option of the answer like Valid or Invalid.
After giving the final answer, STOP immediately.
Do not generate any new question.
Answer:"""

# Output format: (A) or (B) or (C), etc. (no explanation)

# prompt_template_bbh_yes_no = """
# Answer the following question.
# Think carefully and reason step by step internally before answering.
# Return ONLY one word: Yes or No.

# After giving the final answer, STOP immediately. Do not generate any new question. Examples: Yes, No.

# Question: {Q}
# Answer:"""

# prompt_template_bbh_true_false = """
# Answer the following question.
# Think carefully and reason step by step internally before answering.

# Return ONLY one word: True or False.

# Question: {Q}
# Answer:"""

# prompt_template_bbh_valid = """
# Answer the following question.
# Think carefully and reason step by step internally before answering.
# Return ONLY one word: Valid or Invalid.

# After giving the final answer, STOP immediately. Do not generate any new question. Examples: Valid, Invalid.

# Question: {Q}

# Answer:"""
# # Output only: Valid or Invalid (no explanation)

