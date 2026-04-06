import json
import os

import torch

# device = "cuda:0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BPO_MODEL = "THUDM/BPO"
MEPO_MODEL= "zixiaozhu/MePO"

MINILM_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
GEMMA_EMBEDDING_MODEL = "google/embeddinggemma-300m"
embedding_models = [MINILM_EMBEDDING_MODEL, GEMMA_EMBEDDING_MODEL]

distance_thresholds = {
    f'{MINILM_EMBEDDING_MODEL}': 0.05,
    # f'{QWEN_EMBEDDING_MODEL}': 0.1,
    f'{GEMMA_EMBEDDING_MODEL}': 0.14
}

DEEPSEEK = "deepseek-chat"
evaluator_models = [DEEPSEEK]   

LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"
base_llm_models = [VICUNA_7B,LLAMA2_7B, GEMMA3]

DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.json"
BPO_EVAL = "testset/bpo_test.json"
SELF_INSTRUCT_EVAL = "testset/self_instruct_eval.json"
evaluation_datasets = [VICUNA_EVAL, DOLLY_EVAL, BPO_EVAL, SELF_INSTRUCT_EVAL]

mepo_folder_name = "rbpo_mepo"
mismatch_folder_name = "mismatch"
consistency_folder_name = "consistency"
eval_folder_name = "evaluation"
experiment_file_name = "experiment.txt"

BPO = "bpo"
RBPO = "rbpo"
MEPO = "mepo"
RMEPO = "rmepo"
METHOD = [BPO, RBPO, MEPO, RMEPO]

M = 10 # prompt optimization iterations
DISTANCE_THRESHOLD=0.05
IMP_ENC=0.5

def clean_name(path_or_id: str):
    name = path_or_id.split("/")[-1]        
    name = name.split(":")[0]            
    return os.path.splitext(name)[0]     

def create_combined_name(model_path: str, dataset: str, evaluator: str):
    model_name = clean_name(model_path)
    dataset_name = clean_name(dataset)
    evaluator_name = clean_name(evaluator)
    
    model_abbr = model_name.split("-")[0].split("_")[0].lower()
    dataset_abbr = dataset_name.split("-")[0].split("_")[0].lower()
    evaluator_abbr = evaluator_name.split("-")[0].split("_")[0].lower()
    
    return f"{model_abbr}_{dataset_abbr}_{evaluator_abbr}"

def convert_analysis_path_to_figure(path: str, suffix: str = "ori_rbpo") -> str:
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)

    try:
        analysis_idx = parts.index("analysis")

        model = parts[analysis_idx + 1].split("-")[0]
        eval_name = parts[analysis_idx + 2].split("-")[0]
        judge = parts[analysis_idx + 3].split("-")[0]

    except (ValueError, IndexError):
        raise ValueError("Path không đúng cấu trúc src/analysis/...")

    return os.path.join(
        "src",
        "figure",
        f"{model}_{eval_name}_{judge}_{suffix}"
    )

def dataset_processing(file_path: list):
    import os
    os.makedirs(eval_folder_name, exist_ok=True)
    for base_model in base_llm_models:
        for data_path in evaluation_datasets:
            for evaluator in evaluator_models:
                
                file_name = create_combined_name(base_model, data_path, evaluator)
                
                with open(data_path, "r", encoding="utf-8") as f:
                    data =  json.load(f)
                # Chuyển đổi sang định dạng chuẩn: id, ori_prompt, bpo_prompt, bpo_res, rbpo_prompt, rbpo_res, category, expected_response
                
                results = []
                for idx, item in enumerate(data, start=1):
                    results.append({
                        "id": item.get("id") or item.get("question_id") or item.get("idx") or idx,  # giữ nguyên id nếu đã có, nếu không có thì để None
                        "ori_prompt": item.get("instruction", None) or item.get("prompt", None) or item.get("text", None),
                        "context": item.get("context", None),
                        "bpo_prompt": item.get("optimized_prompt", None),
                        "category": item.get("category", None),
                        "expected_response": item.get("output", None) or item.get("good_res", None) or item.get("response", None)
                    })
                
                with open(f"{eval_folder_name}/{file_name}.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
def print_file_names(file_path: str):
    results = []        
    for base_model in base_llm_models:
        for data_path in evaluation_datasets:
            for evaluator in evaluator_models:
                file_name = create_combined_name(base_model, data_path, evaluator)
                results.append(file_name)
    print("\n".join(results))
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    # dataset_processing(evaluation_datasets)
    print_file_names(experiment_file_name)
    