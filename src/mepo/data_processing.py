data_path = ["gemma_dolly_deepseek",
            "gemma_vicuna_deepseek",
            "llama_dolly_deepseek",
            "llama_vicuna_deepseek",
            "vicuna_dolly_deepseek",
            "vicuna_vicuna_deepseek"
            ]

vicuna_data_path = ["gemma_vicuna_deepseek",
                    "llama_vicuna_deepseek",
                    "vicuna_vicuna_deepseek"
                    ]

dolly_data_path = ["gemma_dolly_deepseek",
                    "llama_dolly_deepseek",
                    "vicuna_dolly_deepseek"
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

# for path in data_path:
#     with open(f"{path}.jsonl", "r", encoding="utf-8") as f:
#         data = [json.loads(line) for line in f]
#     for item in data:
#         # Xử lý an toàn - chỉ đổi tên khóa nếu tồn tại
#         if "org_prompt" in item:
#             item["ori_prompt"] = item.pop("org_prompt")
#         if "prompt_0" in item:
#             item["bpo_prompt"] = item.pop("prompt_0")
#         if "res_0" in item:
#             item["bpo_res"] = item.pop("res_0")
#         if "prompt_1" in item:
#             item["rbpo_prompt"] = item.pop("prompt_1")
#         if "res_1" in item:
#             item["rbpo_res"] = item.pop("res_1")
#         # Tất cả các field khác (bao gồm mepo fields) sẽ được giữ nguyên
    
#     with open(f"{path}.json", "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4, ensure_ascii=False)
        
# Vao them id cho tung item
for path in data_path:
    with open(f"{path}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, item in enumerate(data,1):
        item["id"] = idx
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
# Kiem tra xem id moi data co key "ori_prompt" giong nhau hay khong.
# Muc tieu: kiem tra xem data co bi xao tron hay khong, neu id 1 co "ori_prompt" la "abc" thi id 1 o data khac cung phai la "abc"

# with open(f"{dolly_data_path[0]}.json", "r", encoding="utf-8") as f:
#     data0 = json.load(f)
# with open(f"{dolly_data_path[1]}.json", "r", encoding="utf-8") as f:
#     data1 = json.load(f)
# with open(f"{dolly_data_path[2]}.json", "r", encoding="utf-8") as f:
#     data2 = json.load(f)
# idx = 0
# conflict = False
# while idx < len(dolly_data_path[0]):
#     if data0[idx]["ori_prompt"] != data1[idx]["ori_prompt"] or data0[idx]["ori_prompt"] != data2[idx]["ori_prompt"]:
#         print(f"Mismatch at index {idx}:")
#         print(f"Data 0: {data0[idx]['ori_prompt']}")
#         print(f"Data 1: {data1[idx]['ori_prompt']}")
#         print(f"Data 2: {data2[idx]['ori_prompt']}")
#         conflict = True
#     idx += 1

# if not conflict:
#     print("Done!")
# else :
#     print("Sth Wrong!")