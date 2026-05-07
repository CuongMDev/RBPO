
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from config import MODEL_CACHE_PATH
from helper import (LLAMA2_7B, PIQA, clean_name, downstream_folder_name, downstream_tasks, RBPO, RMEPO, GEMMA_EMBEDDING_MODEL, MINILM_EMBEDDING_MODEL, base_llm_models, BBH, device, embedding_models, load_model_and_tokenizer)
import json, os, torch, gc

from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice
from utils import generate, generate_batch
from utils_downstream_task import format_prompt_template

load_dotenv()
hf_token = os.getenv("HF_TOKEN")    

# embedding_models = [
#     GEMMA_EMBEDDING_MODEL,
#     MINILM_EMBEDDING_MODEL
# ]

base_llm_models = [LLAMA2_7B]

METHOD = [
    RBPO,
    RMEPO
]

res = []
for base_model in base_llm_models:
    model, tokenizer = load_model_and_tokenizer(base_model, device_map="auto", cache_dir=MODEL_CACHE_PATH, token=hf_token)
    for method in METHOD:
        for embed_model_name in embedding_models:
            for task in downstream_tasks:
                if task == "demo":
                    continue
                # if task == BBH:
                #     multiple_choice = [
                #     'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
                #     'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                #     'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                #     'ruin_names', 'salient_translation_error_detection', 'snarks',
                #     'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                #     'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                #     'causal_judgement','formal_fallacies','navigate','web_of_lies',
                #     'sports_understanding','boolean_expressions',
                #     'multistep_arithmetic_two', 'object_counting', 'word_sorting'
                # ]
                #     for subtask in multiple_choice:
                #         data_path = f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}/{subtask}.json"
                #         with open(data_path, "r", encoding="utf-8") as f:
                #             data = json.load(f) # NOTE: chỉ lấy 5 sample đầu để test
                #         print(f"\nBase model: {base_model} | Task: {subtask} | Embedding: {clean_name(embed_model_name)} | Dataset: {data_path} | Loaded {len(data)} samples")                        
                        
                #         prompts = []
                #         indices = []
                #         for i, item in enumerate(data):
                #             prompt = format_prompt_template(
                #                 task_name=subtask,
                #                 item=item,
                #                 method_key=method
                #             )
                #             item[f'{method}_input_llm'] = prompt
                #             prompts.append(prompt)
                #             indices.append(i)

                #         responses = generate_batch(
                #             model=model,
                #             tokenizer=tokenizer,
                #             prompts=prompts,
                #             context=None,
                #             do_sample=False,
                #             apply_chat_template=False,
                #             device=device
                #         )
                #         for i, response in zip(indices, responses):
                #             data[i][f"{method}_response"] = response
                #         with open(data_path, "w", encoding="utf-8") as f:
                #             json.dump(data, f, ensure_ascii=False, indent=2)
                # else:
                elif task == PIQA:
                    data_path= f"{downstream_folder_name}/{method}/{clean_name(embed_model_name)}/{task}.json"

                    with open(data_path, "r", encoding="utf-8") as f:
                        data = json.load(f) # NOTE: chỉ lấy 5 sample đầu để test
                    print(f"\nBase model: {base_model} | Task: {task} | Embedding: {clean_name(embed_model_name)} | Loaded {len(data)} samples")
                    
                    prompts = []
                    indices = []
                    for i, item in enumerate(data):
                        prompt = format_prompt_template(
                            task_name=task,
                            item=item,
                            method_key=method
                        )
                        item[f'{method}_input_llm'] = prompt   
                        prompts.append(prompt)
                        indices.append(i)
                        
                    with torch.no_grad():
                        responses = generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompts=prompts,
                            context=None,
                            do_sample=False,
                            apply_chat_template=False,
                            device=device
                        )   
                    for i, response in zip(indices, responses):
                        data[i][f"{method}_response"] = response
                        
                    with open(data_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
    
    del model
    torch.cuda.empty_cache() 
    gc.collect()  
    if os.path.exists(MODEL_CACHE_PATH):
        import shutil
        shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)           