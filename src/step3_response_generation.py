import shutil

from helper import (GEMMA3, GEMMA_EMBEDDING_MODEL, LLAMA2_7B, MEPO, METHOD, RMEPO, SELF_INSTRUCT_EVAL, VICUNA_7B, VICUNA_EVAL, clean_name,
    create_combined_name, eval_folder_name, experiment_file_name, base_llm_models, embedding_models,
    evaluation_datasets,evaluator_models, DOLLY_EVAL, BPO, RBPO, device, BPO_EVAL, SELF_INSTRUCT_EVAL)
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH, prompt_template_vicuna

import os, json, torch, gc, time

from dotenv import load_dotenv

from utils import generate_batch

print("===== STEP 3: Response Generation =====")
torch.cuda.empty_cache()
gc.collect()

load_dotenv()
hf_token = os.getenv("HF_TOKEN")    

embedding_models = [GEMMA_EMBEDDING_MODEL]
# base_llm_models = [VICUNA_7B]
# evaluation_datasets = [BPO_EVAL, SELF_INSTRUCT_EVAL]

for model_name in embedding_models:
    for base_model in base_llm_models:
        torch.cuda.empty_cache(), gc.collect()
        is_vicuna = (base_model is VICUNA_7B)  # nếu là VICUNA_7B thì is_vicuna = True, ngược lại False
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=MODEL_CACHE_PATH,
            token = hf_token,
            torch_dtype="auto"
        ).eval().to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=MODEL_CACHE_PATH,
            token = hf_token,
            legacy=False
        )
        
        for data_path in evaluation_datasets:
            is_need_context = (data_path in [DOLLY_EVAL, SELF_INSTRUCT_EVAL])  # nếu là DOLLY_EVAL hoặc SELF_INSTRUCT_EVAL thì cần context
            
            for evaluator in evaluator_models:
                file_name = create_combined_name(base_model, data_path, evaluator)
                with open(f'{eval_folder_name}/{clean_name(model_name)}/{file_name}.json', "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"\nBase model: {base_model} | Dataset: {data_path} | Evaluator: {evaluator} | Loaded {len(data)} samples")                
                # continue
                
                for item in data:
                    all_prompts = []
                    # ori_prompt = item.get("ori_prompt", "") 
                    bpo_prompt = item.get("bpo_prompt", "")
                    rbpo_prompt = item.get("rbpo_prompt", "")
                    mepo_prompt = item.get("mepo_prompt", "")
                    rmepo_prompt = item.get("rmepo_prompt", "")
                    all_prompts.extend([
                        # ori_prompt,
                        bpo_prompt, 
                        rbpo_prompt,
                        mepo_prompt,
                        rmepo_prompt]
                    )
                    
                    unique_prompts = []
                    prompt_to_idx = {}
                    ori_to_unique = {}
                    
                    for p in all_prompts:
                        if p not in prompt_to_idx:
                            prompt_to_idx[p] = len(unique_prompts)
                            unique_prompts.append(p)
                        ori_to_unique[p] = prompt_to_idx[p]
                    
                    if is_vicuna:
                        unique_prompts = [prompt_template_vicuna.format(p) for p in unique_prompts] # for Vicuna-style model (turn off apply_chat_template)
                                        
                    unique_responses = generate_batch(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=unique_prompts,
                        context=item.get("context", None) if is_need_context else None,
                        do_sample=False,
                        apply_chat_template=False,  # đã format sẵn theo kiểu Vicuna nên tắt apply_chat_template
                        device=device,
                    )
                    
                    responses = [unique_responses[ori_to_unique[p]] for p in all_prompts]
                    
                    # item["ori_response"] = responses[0]
                    item["bpo_response"] = responses[0]
                    item["rbpo_response"] = responses[1]
                    item["mepo_response"] = responses[2]
                    item["rmepo_response"] = responses[3]
                
                with open(f'{eval_folder_name}/{clean_name(model_name)}/{file_name}.json', "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
    
        if os.path.exists(MODEL_CACHE_PATH):
            shutil.rmtree(MODEL_CACHE_PATH, ignore_errors=True)