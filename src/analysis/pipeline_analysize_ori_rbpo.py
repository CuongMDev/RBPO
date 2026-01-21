from data_preprocessing import data_preprocessing
from analysizing_ori_rbpo import classify_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
from openai import OpenAI

path = "src/analysis/"

evaluator = ["claude4/","gemma3/","qwen/"]

folder_names = ["vicuna_llama/",
               "vicuna_vicuna/",
               "dolly_vicuna/",
               "dolly_llama/"]

file_paths = ["lose_pairwise_results_ori_rbpo.jsonl",
             "lose_pairwise_results_bpo_rbpo.jsonl" ]

MODEL_NAME = "xiaomi/mimo-v2-flash:free"
MAX_TOKENS = 2048
TEMPERATURE = 0.0

if __name__ == "__main__":
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Please set it via environment variable."
        )
        
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    for evaluator_name in evaluator:
        folder_names_evaluator = [evaluator_name + folder for folder in folder_names]
        data_preprocessing(folder_names_evaluator, file_paths, path)
        
        input = "lose_pairwise_results_ori_rbpo_preprocessed.json"
        output = "lose_pairwise_results_ori_rbpo_classified.jsonl"
        
        for folder in folder_names_evaluator:
            
            print("Processing folder:", folder)
            INPUT_PATH = os.path.join(folder, input)
            OUTPUT_PATH = os.path.join(folder, output)
            
            print("Input path:", INPUT_PATH)
            print("Output path:", OUTPUT_PATH)
            
            classify_dataset(
                input_json_path=INPUT_PATH,
                output_jsonl_path=OUTPUT_PATH,
                batch_size=2,     # mỗi batch 1 item
                max_workers=2     # 4 request song song
            )
        
        
    



