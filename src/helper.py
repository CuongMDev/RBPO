import os

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
    
if __name__ == "__main__":
    test_path = "src/analysis/llama2-7b-chat-hf/vicuna_eval/deepseek-chat/lose_pairwise_results_ori_rbpo.jsonl"
    figure_path = convert_analysis_path_to_figure(test_path, suffix="test")
    print(figure_path)