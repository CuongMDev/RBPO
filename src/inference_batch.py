BATCH_SIZE= 10
from mepo_inference import MePOModel
from tqdm import tqdm
import json

testset_filename = ["dolly_eval", "vicuna_eval"]
folder_path = "testset"
model = MePOModel()

for data_path in testset_filename:
    result = []

    with open(f"{folder_path}/{data_path}.json", "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(len(corpus))
    
    batch_prompts = []
    batch_refs = []

    for sample in tqdm(corpus):
        ori_prompt = sample.get('instruction') or sample.get('text')
        if not ori_prompt:
            continue

        po_qs_input = model.po_prompt_ins.replace("S_P", ori_prompt)

        batch_prompts.append(po_qs_input)
        batch_refs.append(ori_prompt)

        # chạy khi đủ batch
        if len(batch_prompts) == BATCH_SIZE:
            outputs = model.generate_batch(batch_prompts)

            for ori, opt in zip(batch_refs, outputs):
                result.append({
                    "ori_prompt": ori,
                    "optim_prompt": opt
                })

            batch_prompts = []
            batch_refs = []

    # xử lý batch cuối
    if batch_prompts:
        outputs = model.generate_batch(batch_prompts)
        for ori, opt in zip(batch_refs, outputs):
            result.append({
                "ori_prompt": ori,
                "optim_prompt": opt
            })
    with open(f"{data_path}_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
