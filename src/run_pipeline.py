import os
import torch
from openai import OpenAI
from pipeline_utils import loading_data, run_pairwise_ranking, step1_generate_paraphrase, step2_sbert_clustering, step3_infer_response
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(42)

CLAUDE4 = "anthropic/claude-sonnet-4"
DEEPSEEK = "tngtech/deepseek-r1t2-chimera:free"
LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"
DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.jsonl"
DEMO_EVAL = "testset/demo.json"

# evaluator_models = [CLAUDE4, DEEPSEEK]
# base_llm_models = [LLAMA2_7B, VICUNA_7B, GEMMA3]
# evaluation_datasets = [DOLLY_EVAL, VICUNA_EVAL]
evaluator_models = [DEEPSEEK]
base_llm_models = [LLAMA2_7B]
evaluation_datasets = [DEMO_EVAL]

for evaluator in evaluator_models:
    # Vào folder của mỗi evaluator
    os.makedirs(f"results/{evaluator.replace('/', '_')}", exist_ok=True)
    for dataset in evaluation_datasets:
        # Vào folder của mỗi dataset
        os.makedirs(f"results/{evaluator.replace('/', '_')}/{os.path.basename(dataset).replace('.', '_')}", exist_ok=True)
        for base_model in base_llm_models:
            # Vào folder của mỗi base model
            os.makedirs(f"results/{evaluator.replace('/', '_')}/{os.path.basename(dataset).replace('.', '_')}/{base_model.replace('/', '_')}", exist_ok=True)
            print(f"\n=== Evaluator: {evaluator} | Dataset: {dataset} | Base model: {base_model} ===\n")
            
            # Chạy pipeline
            loading_data(input_path=dataset)
            step1_generate_paraphrase()
            # step2_sbert_clustering()
            # step3_infer_response(infer_model_path=base_model, is_vicuna=False)
            # print("\n🎉 ALL DONE!")
            # run_pairwise_ranking(evaluator=evaluator)
