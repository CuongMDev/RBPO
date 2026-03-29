data_path = ["gemma_dolly_deepseek",
            "gemma_vicuna_deepseek",
            "llama_dolly_deepseek",
            "llama_vicuna_deepseek",
            "vicuna_dolly_deepseek",
            "vicuna_vicuna_deepseek"
            ]

import json
import os
import gc
import torch

# Di vao tung data_path.jsonl. doi cac key 
# org_prompt -> original_prompt
# prompt_0 -> method1_prompt
# res_0 -> method1_response
# prompt_1 -> method2_prompt
# res_1 -> method2_response

# chuyen sang json

for path in data_path:
    with open(f"{path}.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    for item in data:
        # Xử lý an toàn - chỉ đổi tên khóa nếu tồn tại
        if "org_prompt" in item:
            item["ori_prompt"] = item.pop("org_prompt")
        if "prompt_0" in item:
            item["bpo_prompt"] = item.pop("prompt_0")
        if "res_0" in item:
            item["bpo_res"] = item.pop("res_0")
        if "prompt_1" in item:
            item["rbpo_prompt"] = item.pop("prompt_1")
        if "res_1" in item:
            item["rbpo_res"] = item.pop("res_1")
    
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)