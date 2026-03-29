import gc
import os
import torch
from openai import OpenAI
from clean_cache import nuke_hf_cache
from config import MODEL_CACHE_PATH
from inference_batch import infer_mepo_res_batch, process_mepo_batch, save_mepo_results, merge_mepo_with_original
from mepo_inference import MePOModel
from pipeline_utils import loading_data, run_pairwise_ranking, step1_generate_paraphrase, step2_sbert_clustering, step3_infer_response
from helper import evaluation_datasets, evaluator_models, base_llm_models, create_combined_name, clean_name, GEMMA3, DEEPSEEK, VICUNA_EVAL, DEMO_EVAL
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

# # Demo
# base_llm_models = [GEMMA3]
# evaluator_models = [DEEPSEEK]
# evaluation_datasets = [VICUNA_EVAL]

device = "cuda:0"
RESULTS_ROOT = "mepo"

#  =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    root = "mepo"
    mepo_model = MePOModel()
    os.makedirs(root, exist_ok=True)
    
    for base_model in base_llm_models:
        torch.cuda.empty_cache(), gc.collect()
        
        if base_model is VICUNA_EVAL:  # nếu là VICUNA_7B
            is_vicuna = True
        else:
            is_vicuna = False
            
        model_infer_res = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=MODEL_CACHE_PATH,
            torch_dtype="auto"
        ).eval().to(device)

        tokenizer_infer_res = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=MODEL_CACHE_PATH,
            legacy=False
        )

        for data_path in evaluation_datasets:
            torch.cuda.empty_cache(), gc.collect()

            for evaluator in evaluator_models:
                torch.cuda.empty_cache(), gc.collect()
                file_path = create_combined_name(base_model, data_path, evaluator)
                file_name = os.path.join(root, f"{file_path}.jsonl")
                # os.makedirs(run_dir, exist_ok=True)
                
                print(f"=== Base model: {clean_name(base_model)} | Dataset: {clean_name(data_path)} | Evaluator: {clean_name(evaluator)} ===")
                 # -------------------------
                # FILE PATHS
                # -------------------------
                # Step1: infer optimized prompt MePO
                result, _, _, _ = process_mepo_batch(base_model, data_path, evaluator, mepo_model)
                                
                # Step3: Merge MePO result với file JSON gốc (có ori_prompt, bpo_prompt, bpo_res, rbpo_prompt, rbpo_res)
                
                # result = [{
                #     "ori_prompt": "How can I improve my time management skills?",
                #     "mepo_prompt": "What is Retrieval-Augmented Generation (RAG)?"
                # }]
                if os.path.exists(file_name):
                    merged_data = merge_mepo_with_original(result, file_name)
                    print(f"✓ Step2: Merged {len(merged_data)} items")
                else:
                    print(f"⚠ File not found: {file_name}") 
                    
                # Step 2: infer response từ optimized prompt MePO
                # output_mepo_response = f'{file_path}_responses.jsonl'
                infer_mepo_res_batch(
                    model = model_infer_res,
                    tokenizer=tokenizer_infer_res,
                    file_path=file_name,
                    device=device,
                    batch_size=10
                )
                nuke_hf_cache(MODEL_CACHE_PATH)
                
                # run_pairwise_ranking(
                #     evaluator=evaluator,
                #     input_path=output_path,
                #     output_jsonls=output_jsonl,
                #     output_dir=run_dir
                # )

    # =========================
    # FINAL CLEANUP
    # =========================
    torch.cuda.empty_cache()
    gc.collect()

    print("\n🎉 ALL EXPERIMENTS DONE")
