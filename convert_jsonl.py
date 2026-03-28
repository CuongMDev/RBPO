import json
import os

# Danh sách các file .jsonl cần convert
jsonl_files = [
    'src/testset/vicuna_eval.jsonl'
]

for jsonl_file in jsonl_files:
    if not os.path.exists(jsonl_file):
        print(f"[SKIP] {jsonl_file} không tồn tại")
        continue
    
    json_file = jsonl_file.replace('.jsonl', '.json')
    
    print(f"[CONVERT] {jsonl_file} → {json_file}")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [ERROR] Lỗi dòng {line_num}: {e}")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Đã convert {len(data)} records")

print("\n✓ Hoàn tất!")
