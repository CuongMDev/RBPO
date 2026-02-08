from data_preprocessing import data_preprocessing
from analysizing_ori_rbpo import classify_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
from openai import OpenAI

file_paths = ["lose_pairwise_results_ori_rbpo.jsonl",
            #  "lose_pairwise_results_bpo_rbpo.jsonl" 
             ]

DEEPSEEK_EVALUATOR = "deepseek-chat"

LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
VICUNA_7B = "lmsys/vicuna-7b-v1.3"
GEMMA3 = "google/gemma-3-4b-it"
DOLLY_EVAL = "testset/dolly_eval.json"
VICUNA_EVAL = "testset/vicuna_eval.jsonl"
DEMO_EVAL = "testset/demo.json"

evaluator_models = [DEEPSEEK_EVALUATOR]
base_llm_models = [LLAMA2_7B, VICUNA_7B, GEMMA3]
base_llm_models = [VICUNA_7B]
evaluation_datasets = [VICUNA_EVAL, DOLLY_EVAL]

def clean_name(path_or_id: str):
    name = path_or_id.split("/")[-1]      # lấy phần sau /
    name = name.split(":")[0]            # bỏ phần sau :
    return os.path.splitext(name)[0]     # bỏ .json / .jsonl nếu có


if __name__ == "__main__":
    
    # load_dotenv()

    # OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    # if not OPENROUTER_API_KEY:
    #     raise RuntimeError(
    #         "OPENROUTER_API_KEY is not set. "
    #         "Please set it via environment variable."
    #     )
        
    # client = OpenAI(
    #     api_key=OPENROUTER_API_KEY,
    #     base_url="https://openrouter.ai/api/v1"
    # )
    
    for base_model in base_llm_models:
        base_dir = clean_name(base_model)
        for eval_dataset in evaluation_datasets:
            eval_dir = os.path.join(base_dir,clean_name(eval_dataset))
            for evaluator in evaluator_models:
                eval_dir = os.path.join(eval_dir,clean_name(evaluator))
                print("Processing directory:", eval_dir)
                for file_path in file_paths:
                    input_path = os.path.join(eval_dir, file_path)
                    output_path = os.path.join(
                        eval_dir,
                        file_path.replace(".jsonl", "_preprocessed.json")
                    )
                    data_preprocessing(
                        input_jsonl_path=input_path,
                        output_json_path=output_path
                    )
                    
                    classify_dataset(
                        input_json_path=output_path,
                        output_json_path=output_path.replace("_preprocessed.json","_classified.json"),
                        batch_size=2,     # mỗi batch 1 item
                        max_workers=2     # 4 request song song
                    )
    # for evaluator_name in evaluator:
    #     folder_names_evaluator = [evaluator_name + folder for folder in folder_names]        
    #     input = "lose_pairwise_results_ori_rbpo_preprocessed.json"
    #     output = "lose_pairwise_results_ori_rbpo_classified.jsonl"
        
    #     for folder in folder_names_evaluator:
            
    #         print("Processing folder:", folder)
    #         INPUT_PATH = os.path.join(folder, input)
    #         OUTPUT_PATH = os.path.join(folder, output)
            
    #         print("Input path:", INPUT_PATH)
    #         print("Output path:", OUTPUT_PATH)
            
    #         classify_dataset(
    #             input_json_path=INPUT_PATH,
    #             output_json_path=OUTPUT_PATH,
    #             batch_size=2,     # mỗi batch 1 item
    #             max_workers=2     # 4 request song song
    #         )
        
        
    



