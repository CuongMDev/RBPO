import json

jsonl_file = 'src/verify_winner/Llama-2-7b-chat-hf/vicuna_eval/ori_rbpo.jsonl'
json_file = 'src/verify_winner/Llama-2-7b-chat-hf/vicuna_eval/ori_rbpo.json'

data = []
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Loại bỏ các dòng trống nếu có và nạp JSON
        if line.strip():
            data.append(json.loads(line))

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)