import gc
import os
import torch
from openai import OpenAI
from clean_cache import nuke_hf_cache
from config import MODEL_CACHE_PATH
from pipeline_utils import loading_data, run_pairwise_ranking, step1_generate_paraphrase, step2_sbert_clustering, step3_infer_response
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

DEEPSEEK = "deepseek-chat"

LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"

DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.jsonl"
DEMO_EVAL = "testset/demo.json"

evaluator_models = [DEEPSEEK]
base_llm_models = [LLAMA2_7B, VICUNA_7B, GEMMA3]
evaluation_datasets = [VICUNA_EVAL, DOLLY_EVAL]
base_llm_models = [LLAMA2_7B]
# base_llm_models = [VICUNA_7B, GEMMA3]
evaluator_models = [DEEPSEEK]

# evaluation_datasets = [DEMO_EVAL]

# =========================
# RUN CONFIG
# =========================
device = "cuda:0"
RESULTS_ROOT = "results"

# =========================
# MODEL LOAD (BPO – dùng cho Step 1)
# =========================
optimize_path = "THUDM/BPO"

# print("🔹 Loading BPO model once...")
# bpo_model = AutoModelForCausalLM.from_pretrained(
#     optimize_path,
#     cache_dir=MODEL_CACHE_PATH,
#     torch_dtype=torch.float16
# ).eval().to(device)

# bpo_tokenizer = AutoTokenizer.from_pretrained(
#     optimize_path,
#     cache_dir=MODEL_CACHE_PATH,
#     use_fast=False,
#     legacy=True
# )
# bpo_model.config.return_dict = True

# if bpo_tokenizer.pad_token_id is None:
#     bpo_tokenizer.pad_token_id = bpo_tokenizer.eos_token_id

def clean_name(path_or_id: str):
    name = path_or_id.split("/")[-1]      # lấy phần sau /
    name = name.split(":")[0]            # bỏ phần sau :
    return os.path.splitext(name)[0]     # bỏ .json / .jsonl nếu có

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    for base_model in base_llm_models:
        # nuke_hf_cache(MODEL_CACHE_PATH) # Dùng khi chạy lại step 1, load model BPO
        
        # if base_model is base_llm_models[1]:  # nếu là VICUNA_7B
        #     is_vicuna = True
        # else:
        #     is_vicuna = False
        
        base_llm_dir = os.path.join(RESULTS_ROOT, clean_name(base_model))
        os.makedirs(base_llm_dir, exist_ok=True)
        
        # model = AutoModelForCausalLM.from_pretrained(
        #     base_model,
        #     cache_dir=MODEL_CACHE_PATH,
        #     torch_dtype=torch.float16
        # ).eval().to(device)

        # tokenizer = AutoTokenizer.from_pretrained(
        #     base_model,
        #     cache_dir=MODEL_CACHE_PATH,
        #     legacy=False
        # )

        for dataset in evaluation_datasets:
            torch.cuda.empty_cache(), gc.collect()

            dataset_name = clean_name(dataset)
            dataset_dir = os.path.join(base_llm_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            for evaluator in evaluator_models:
                torch.cuda.empty_cache(), gc.collect()
                
                evaluator_name = clean_name(evaluator)
                run_dir = os.path.join(dataset_dir, evaluator_name)
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n=== Base model: {base_model} | Dataset: {dataset} | Evaluator: {evaluator_name} ===")

                # -------------------------
                # FILE PATHS
                # -------------------------
                optimized_path = os.path.join(run_dir, "optimized_prompts.jsonl")
                tmp_step1 = os.path.join(run_dir, "tmp_step1_r0.jsonl")
                tmp_step2 = os.path.join(run_dir, "tmp_step2_r0.jsonl")
                output_path = os.path.join(run_dir, "responses_with_semantic.jsonl")

                # -------------------------
                # PIPELINE
                # -------------------------
                # loading_data(
                #     input_path=dataset,
                #     output_path=optimized_path
                # )

                # step1_generate_paraphrase(
                #     model=bpo_model,
                #     tokenizer=bpo_tokenizer,
                #     input_path=optimized_path,
                #     tmp_step1=tmp_step1,
                #     device=device
                # )

                # step2_sbert_clustering(
                #     tmp_step1=tmp_step1,
                #     tmp_step2=tmp_step2,
                #     device=device
                # )

                # step3_infer_response(
                #     model = model,
                #     tokenizer=tokenizer,
                #     tmp_step2=tmp_step2,
                #     output_jsonl=output_path,
                #     device=device,
                #     is_vicuna=is_vicuna
                # )
                
                print(output_path)
                print(run_dir)
                
                ori_bpo_res = os.path.join(run_dir, "lose_pairwise_results_ori_bpo.jsonl")
                ori_rbpo_res = os.path.join(run_dir, "lose_pairwise_results_ori_rbpo.jsonl")
                bpo_rbpo_res = os.path.join(run_dir, "lose_pairwise_results_bpo_rbpo.jsonl")
                output_jsonl = [ori_bpo_res, ori_rbpo_res, bpo_rbpo_res]

                run_pairwise_ranking(
                    evaluator=evaluator,
                    input_path=output_path,
                    output_jsonls=output_jsonl,
                    output_dir=run_dir
                )

                print(f"✓ Run complete → {run_dir}")

    # =========================
    # FINAL CLEANUP
    # =========================
    nuke_hf_cache(MODEL_CACHE_PATH)
    print("\n🎉 ALL EXPERIMENTS DONE")
