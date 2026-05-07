import os
import regex
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from config import MODEL_CACHE_PATH
from tqdm import tqdm
from helper import (MEPO_MODEL, METHOD, RBPO, RMEPO, downstream_tasks,
    downstream_folder_name, M, temp_po_models)
from utils import generate_batch, make_prompt_template
from helper import MEPO, BPO_MODEL,BPO
from config import prompt_template_optimize

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")    

def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().strip()

def remove_non_characters(text):
    # Keep only Unicode letters, numbers, whitespace, and common punctuation
    cleaned_text = regex.sub(r"[^\p{L}\p{N}\p{P}\p{Z}]", "", text)
    return cleaned_text

def load_model_and_tokenizer(model_path,
    device_map="auto",
    cache_dir=MODEL_CACHE_PATH,
    token=HF_TOKEN):
    model = AutoModelForCausalLM.from_pretrained(model_path, 
        device_map=device_map,
        cache_dir=cache_dir,
        token=token,
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=MODEL_CACHE_PATH,
        token=HF_TOKEN,
        legacy=False,
        truncation_side='left',
        padding_side='left'
    )
    model.config.return_dict = True
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    return model, tokenizer

def generate_multi_response(model,
    tokenizer,
    prompt,
    M,
    temperature,
    # is_apply_chat_template=True,
    device="cuda"
):
    # print(f"Generate multi response function")
    prompts = [prompt] * M
    
    messages = [
        [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": p}]
        for p in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    
    # ===== Tokenize =====
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    # Only keep input_ids and attention_mask for generation
    gen_inputs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"]
    }

    gen_outputs  = model.generate(
        **gen_inputs,
        max_new_tokens=512,
        do_sample=True,             
        temperature=temperature,
        top_p=0.9,
        return_dict_in_generate=True
    )    
    sequences = gen_outputs.sequences

    # ===== Correct slicing (important) =====
    input_lengths = (model_inputs.input_ids != tokenizer.pad_token_id).sum(dim=1)

    outputs = [
        seq[input_len:] for seq, input_len in zip(sequences, input_lengths)
    ]
    # ===== Decode =====
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def arc(model2, tokenizer2,downstream_dataset,method_key, save_file, M, device="cuda"):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset['test']['question'])), desc="Processing Answers"):
        prompt = downstream_dataset['test']['question'][i]
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        # Sử dụng batching để sinh M variations cùng lúc
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )
        assert len(batch_prompt) == len(po_opt_prompts), "Mismatch between batch size and generated paraphrases"

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            "choices": downstream_dataset['test']['choices'][i],
            "answer": downstream_dataset['test']['answerKey'][i],
        })
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def BBH(model2, tokenizer2,downstream_dataset,method_key, save_file, M, device="cuda"):
    po_opt = []
    
    # Handle nested structure with 'examples' key
    examples = downstream_dataset.get('examples', downstream_dataset) if isinstance(downstream_dataset, dict) and 'examples' in downstream_dataset else downstream_dataset

    for i in tqdm(range(len(examples)), desc="Processing Answers"):
        prompt = examples[i]['question']

        # Optimize prompt with PO model sử dụng batching
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )
        assert len(batch_prompt) == len(po_opt_prompts), "Mismatch between batch size and generated paraphrases"
        

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            "target": examples[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def BBH_math(model2, tokenizer2,downstream_dataset,method_key, save_file, M, device="cuda"):
    po_opt = []
    
    # Handle nested structure with 'examples' key
    examples = downstream_dataset.get('examples', downstream_dataset) if isinstance(downstream_dataset, dict) and 'examples' in downstream_dataset else downstream_dataset

    for i in tqdm(range(len(examples)), desc="Processing Answers"):
        prompt = examples[i]['input']
        
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )
        assert len(batch_prompt) == len(po_opt_prompts), "Mismatch between batch size and generated paraphrases"

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            "target": examples[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def BBH_wordsorting(model2, tokenizer2,downstream_dataset,method_key, save_file, M, device="cuda"):
    po_opt = []
    
    # Handle nested structure with 'examples' key
    examples = downstream_dataset.get('examples', downstream_dataset) if isinstance(downstream_dataset, dict) and 'examples' in downstream_dataset else downstream_dataset

    for i in tqdm(range(len(examples)), desc="Processing Answers"):
        prompt = examples[i]['input']
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            "target": examples[i]['target'],
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def gsm8k(model2, tokenizer2,downstream_dataset,method_key, save_file, M, device="cuda"):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset['test']['question'])), desc="Processing Answers"):
        prompt = downstream_dataset['test']['question'][i]
        
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            "answer": downstream_dataset['test']['answer'][i],
        })
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)

def piqa(model2, tokenizer2,downstream_dataset, method_key, save_file, M, device="cuda"):
    po_opt = []

    for i in tqdm(range(len(downstream_dataset)), desc="Processing Answers"):
        prompt = downstream_dataset[i]['goal']
        batch_prompt = [prompt_template_optimize.format(prompt) for _ in range(M)]
        po_opt_prompts = generate_batch(model2,
            tokenizer2,
            batch_prompt,
            temperature=temp_po_models[method_key],
            apply_chat_template=False,
            device=device
        )

        po_opt.append({
            'raw_question': prompt,
            f"{method_key}_rephrases": po_opt_prompts,
            'sol1': downstream_dataset[i]['sol1'],
            'sol2': downstream_dataset[i]['sol2'],
            'label':  downstream_dataset[i]['label'],
            'goal':  downstream_dataset[i]['goal']
        })

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(po_opt, f, indent=4, ensure_ascii=False)
            
def read_json(file_path):
    """Reads a JSON file and returns the parsed data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data
    return data

with open("./optimize_prompt_instruction.txt", "r",
        encoding="utf-8") as file:
    po_prompt_ins = file.read()
    
def inspect_dataset(name, dataset, n=2):
    print(f"\n===== {name} =====")
    
    if isinstance(dataset, dict):  # huggingface DatasetDict
        for split in dataset:
            print(f"\n--- Split: {split} ---")
            print("Columns:", dataset[split].column_names)
            print("Size:", len(dataset[split]))
            print("Sample:")
            for i in range(min(n, len(dataset[split]))):
                print(dataset[split][i])
    else:  # json list
        print("Size:", len(dataset))
        print("Sample:")
        for i in range(min(n, len(dataset))):
            print(dataset[i])
    
print("Loading PO model...")
model2, tokenizer2 = load_model_and_tokenizer(BPO_MODEL)

# downstream_tasks = ["demo"]
METHOD = [RBPO]

for method_key in METHOD:
    for task in downstream_tasks:
        import torch, gc
        print('='*80)
        print(torch.cuda.empty_cache(),gc.collect())
        print(f"\nProcessing task: {task} with method: {method_key}")
        
        save_file = f"{downstream_folder_name}/{method_key}/{task}_optim.json"
        if not os.path.exists(save_file):
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
        if task.lower()=='demo':
            downstream_dataset = read_json(f'./testset/demo.json')
            piqa(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        # if task.lower()=='arc_easy':
        #     downstream_dataset=load_dataset('allenai/ai2_arc', 'ARC-Easy')
        #     arc(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        # elif task.lower() == 'arc_challenge':
        #     downstream_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        #     arc(model2, tokenizer2,downstream_dataset,method_key,  save_file, M)
        # elif task.lower()=='gsm8k':
        #     downstream_dataset = load_dataset("gsm8k", "main")
        #     gsm8k(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        elif task.lower() == 'piqa':
            downstream_dataset = read_json( f'./testset/{task}.json')
            piqa(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        # elif task.upper() == 'BBH':
        #     multiple_choice = [
        #         # ARC_C,
        #         # ARC_E,
        #         'snarks',
        #         'ruin_names', 
        #         'hyperbaton', 
        #         'movie_recommendation',
        #         'penguins_in_a_table',
        #         'temporal_sequences',       
        #         'date_understanding',
        #         'disambiguation_qa', 
        #         'logical_deduction_three_objects',
        #         'logical_deduction_five_objects',
        #         'logical_deduction_seven_objects',
        #         'reasoning_about_colored_objects',
        #         'salient_translation_error_detection',
        #         'tracking_shuffled_objects_five_objects', 
        #         'tracking_shuffled_objects_seven_objects',
        #         'tracking_shuffled_objects_three_objects'
        #     ] # NOTE: select between Multiple choices, ARC
        #     # [
        #     #     'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
        #     #     'logical_deduction_seven_objects', 'logical_deduction_three_objects',
        #     #     'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
        #     #     'ruin_names', 'salient_translation_error_detection', 'snarks',
        #     #     'temporal_sequences', 'tracking_shuffled_objects_five_objects',
        #     #     'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
        #     #     'causal_judgement','formal_fallacies','navigate','web_of_lies',
        #     #     'sports_understanding','boolean_expressions',
        #     #     'multistep_arithmetic_two', 'object_counting', 'word_sorting'
        #     # ]
        #     for subtask in multiple_choice:
        #         save_file = f"{downstream_folder_name}/{method_key}/{task}/{subtask}_optim.json"
        #         if not os.path.exists(save_file):
        #             os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
        #         downstream_dataset =read_json(f'./testset/{task}/{subtask}.json')
        #         if subtask == 'word_sorting':
        #             BBH_wordsorting(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        #         elif subtask in ['multistep_arithmetic_two', 'object_counting']:
        #             BBH_math(model2, tokenizer2,downstream_dataset, method_key, save_file, M)
        #         else:
        #             BBH(model2, tokenizer2,downstream_dataset, method_key, save_file, M)


    