import os

DEEPSEEK = "deepseek-chat"

LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"

DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.json"
DEMO_EVAL = "testset/demo.json"

evaluator_models = [DEEPSEEK]
base_llm_models = [VICUNA_7B,LLAMA2_7B, GEMMA3]
evaluation_datasets = [VICUNA_EVAL, DOLLY_EVAL]

output_mepo_folder = "mepo"


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
    
# if __name__ == "__main__":
#     test_path = "src/analysis/llama2-7b-chat-hf/vicuna_eval/deepseek-chat/lose_pairwise_results_ori_rbpo.jsonl"
#     figure_path = convert_analysis_path_to_figure(test_path, suffix="test")
#     print(figure_path)