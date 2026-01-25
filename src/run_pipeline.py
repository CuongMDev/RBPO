import os
import torch
from openai import OpenAI
from pipeline_utils import loading_data, run_pairwise_ranking, step1_generate_paraphrase, step2_infer_vicuna, step3_sbert_clustering
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH


torch.manual_seed(42)

CLAUDE4 = "anthropic/claude-sonnet-4"
DEEPSEEK = "tngtech/deepseek-r1t2-chimera:free"
LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"
DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.jsonl"

evaluator_models = [CLAUDE4, DEEPSEEK]
base_llm_models = [LLAMA2_7B, VICUNA_7B, GEMMA3]
evaluation_datasets = [DOLLY_EVAL, VICUNA_EVAL]

model_path = 'THUDM/BPO'
device = 'cuda:0'

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH, use_fast=False)
model.config.return_dict = True

# Nếu pad_token chưa set, set = eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

loading_data(input_path="testset/demo.json")
step1_generate_paraphrase()
step2_infer_vicuna(base_llm_models[0])
step3_sbert_clustering()
# print("\n🎉 ALL DONE!")
run_pairwise_ranking()
