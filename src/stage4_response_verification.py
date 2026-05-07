import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from helper import (ARC_C, ARC_E, GSM8K, LLAMA2_7B, PIQA, clean_name, downstream_folder_name,eval_folder_name, downstream_tasks, RBPO, RMEPO, GEMMA_EMBEDDING_MODEL, MINILM_EMBEDDING_MODEL, base_llm_models, BBH, device, embedding_models, DEEPSEEK)
import json, os

from config import prompt_template_gsm8k, prompt_template_piqa, prompt_template_multiple_choice
from utils import generate, generate_batch
from utils_downstream_task import format_prompt_template, keyword_verify, llm_verify

folder_path = f"{eval_folder_name}/{downstream_folder_name}"

base_llm_models = [LLAMA2_7B]

embedding_models = [
    # GEMMA_EMBEDDING_MODEL,
    MINILM_EMBEDDING_MODEL
]

METHOD = [
    # RBPO,
    RMEPO
]
        
res = []
for method in METHOD:
    for base_model in base_llm_models:
        for embed_model_name in embedding_models:
            for task in downstream_tasks:
                if task == "demo":
                    continue
                elif task == PIQA:
                    continue
                elif task == BBH:
                    # continue
                    multiple_choice = [
                    'date_understanding', 'disambiguation_qa', 'hyperbaton', 'logical_deduction_five_objects',
                    'logical_deduction_seven_objects', 'logical_deduction_three_objects',
                    'movie_recommendation', 'penguins_in_a_table', 'reasoning_about_colored_objects',
                    'ruin_names', 'salient_translation_error_detection', 'snarks',
                    'temporal_sequences', 'tracking_shuffled_objects_five_objects',
                    'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
                    'causal_judgement','formal_fallacies','navigate','web_of_lies',
                    'sports_understanding','boolean_expressions',
                    'multistep_arithmetic_two', 'object_counting', 
                    'word_sorting'
                ]
                    for subtask in multiple_choice:
                        data_path = f"{folder_path}/{method}/{clean_name(embed_model_name)}/{task}/{subtask}.json"
                        print(f"\nBase model: {base_model} | Task: {subtask} | Embedding: {clean_name(embed_model_name)} | Dataset: {data_path}")                        
                        
                        # keyword_verify(data_path, subtask, method_key=method, is_bbh=True)
                        llm_verify(data_path, subtask, method_key=method, is_bbh=True)
                else:
                    # continue
                # elif task == GSM8K:
                    data_path= f"{folder_path}/{method}/{clean_name(embed_model_name)}/{task}.json"
                    print(f"\nBase model: {base_model} | Task: {task} | Embedding: {clean_name(embed_model_name)} | Dataset: {data_path}")
                    
                    # keyword_verify(data_path, task, method_key=method, is_bbh=False)
                    llm_verify(data_path, task, method_key=method, is_bbh=False)
                    