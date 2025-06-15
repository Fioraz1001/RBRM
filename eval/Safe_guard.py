from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
from tqdm import tqdm
parser =  argparse.ArgumentParser()

parser.add_argument('--data_path',default=None)
parser.add_argument('--write_path')

args = parser.parse_args()

file_path = args.data_path
write_path = args.write_path

model_id = "../model/Meta-Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

dialogs = []
items = []
fp = open(file_path,'r',encoding='utf-8')
idx = 0
cnt = 0
test_idx = 0
for line in fp.readlines():
    if cnt < test_idx:
        cnt += 1
        continue
    dt = json.loads(line)
    # content_0 = "Please respond to my input actively and cooperatively." + "input:" + dt['question'] + "output:"
    content_0 = [
        {"role": "user", "content": dt['input']},
        {"role": "assistant", "content": dt['output']},
    ]
    # content_0 = dt['input'] + "Assistant:"
    dialogs.append(content_0)
    items.append(dt)
    cnt += 1
    # if cnt >= test_idx + length:
    #     break

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)


def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

print(len(dialogs))

results = []
for i,item in tqdm(enumerate(dialogs)):
    result = moderate(item)
    # item["Safe Guard"] = result
    results.append([{"tag":result}])
    items[i]["Safe Guard"] = result

# with open(write_path, 'w', encoding='utf-8') as fw:
#     # 将每个字典转换为JSON格式的字符串，并用换行符分隔
#     for rss in results:
#         fw.writelines(json.dumps(item, ensure_ascii=False) +'\n' for item in rss)

with open(write_path, 'w', encoding='utf-8') as fw:
    fw.writelines(json.dumps(item, ensure_ascii=False) +'\n' for item in items)
