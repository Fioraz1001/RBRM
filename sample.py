from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
import argparse
import csv
import pandas as pd
import numpy as np
import re
from datasets import load_dataset
from tqdm import tqdm

print(torch.__version__)

if torch.cuda.is_available():
    device = torch.device(0)
    print("here:",device)

# file_path = "test_sft_alpaca_preference.jsonl"
# model_dir = "./llama-hf"
# adapter_path = "./lora-alpaca-preferencr-sft/"

def find_first_digit(string):
    match = re.search(r'[12]', string)
    if match:
        return match.group()
    else:
        return None

parser =  argparse.ArgumentParser()
parser.add_argument('--gpus', default="0,1,2,3,4,5,6,7", type=str)

parser.add_argument('--data_path',type=str)
parser.add_argument('--model_path',type=str)
parser.add_argument('--lora_path',default=None,type=str)
parser.add_argument('--hidden_states',default=None)
parser.add_argument('--write_path')

debug = False


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# print(os.environ["CUDA_VISIBLE_DEVICES"],os.environ["WORLD_SIZE"])

file_path = args.data_path
model_dir = args.model_path
adapter_path = args.lora_path
write_path = args.write_path


print("data_path:",file_path)
print("model_path:",model_dir)

generation_config = dict(
    do_sample=True,
    # num_beams=4,
    top_k = 8,
    top_p = 0.95,
    temperature=1.4,
    repetition_penalty=1.2,
    output_hidden_states=False,
    return_dict_in_generate=False,
    num_return_sequences=8,
    max_new_tokens=256
    )


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

TemplateLora = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
Inst = "Respond to this input:"
# Inst = "Please respond to my input actively and cooperatively.input:"

def PromptTemplate(sys_prompt,dialogs,tail = None):
    ids = 0
    dialog_list = []
    if tail is None:
        for dialog in dialogs:
            dialog_list.append(f"<s>{B_INST}{B_SYS}{sys_prompt}{E_SYS}{dialog}{E_INST}")
            ids = ids + 1
    else:
        for dialog in dialogs:
            dialog_list.append(sys_prompt + dialog + tail)
            ids = ids + 1
    return dialog_list


model = AutoModelForCausalLM.from_pretrained(model_dir,device_map='auto')
print("==============================================")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
dialogs = [] 
tag_list = []

# reward_name = "../model/reward-model-deberta-v3-large-v2"
# rank_model, rank_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)


if adapter_path is not None:
    model.load_adapter(adapter_path)
    model.enable_adapters()
    print("lora_path:",adapter_path)

# #################################
# # dataset = load_dataset(file_path,data_dir="red-team-attempts")
# dataset = load_dataset(file_path,data_dir="harmless-base")

# print(dataset)
# print(dataset['train'][0])
# for line in dataset['train']:
#     content = line['chosen'] + '</s>'
#     dialogs.append(content)

# #################################

print(model)
print("==============================================")
max_batch_size = 128

# sys_prompt = "Which response of the following statement is safer?"

# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)
# for idx,dt in enumerate(data):
#     print(dt["instruction"])
#     prompt = Inst + dt["instruction"] + 'output :'
#     label = dt['output']
#     dialogs.append({"idx":idx, "prompt":prompt, "label":label})
#     # if idx >= 5:
#     #     break

fp = open(file_path,'r',encoding='utf-8')
idx = 0
for line in fp.readlines():
    dt = json.loads(line)

    if 'prompt' in dt.keys():
        content_0 = Inst + dt["prompt"] + 'output :'
    elif 'input' in dt.keys():
        content_0 = Inst + dt["input"] + 'output :'

    # content_0 = Inst + dt["prompt"] + 'output :'
    dialogs.append({"idx":idx,"input":content_0})
    idx = idx + 1
    if idx >= 4000:
        break


# dialogs = PromptTemplate(sys_prompt,dialogs,"The safer response is response_")
datasize = len(dialogs)
print(datasize)
results = []
results_hs = []

print(model.device)
print("Start inference.")
fw = open(write_path, 'w', encoding='utf-8')
for index, example in tqdm(enumerate(dialogs)):
    
    input_text = example['input']
    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
    if debug == True:
        results_hs.append(inputs["input_ids"])
        # print("len>",inputs["input_ids"].shape)
        # print(inputs["input_ids"])
        # print(tokenizer.eos_token_id,tokenizer.pad_token_id)
    # print(inputs["input_ids"].shape)
    generation_output = model.generate(
        input_ids = inputs["input_ids"].to(device),
        attention_mask = inputs['attention_mask'].to(device),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        
        **generation_config
    )
    # tmp = []

    print(generation_output)
    for s in generation_output:
        output = tokenizer.decode(s,skip_special_tokens=True)
        
        l = len(example['input'])
        example['output'] = output[l:]
        que = example['input']
        ans = output[l:]
        # ids = rank_tokenizer(que, ans, return_tensors='pt')
        # score = rank_model(**ids).logits[0].cpu().detach().item()
        # example['score'] = score
        print(example)
        fw.write(json.dumps(example) +'\n')
        # results.append(tmp)
        
    # exit()
    if debug: exit()    
    # tmp_list = generation_output.hidden_states
fw.close()
# hidden state position======================================================
    # print("========================================================================")
    # print(len(generation_output.hidden_states[1]))
    # rl = []
    # for itm in generation_output.hidden_states[1]:
    #     rl.append(len(itm))
    # print(rl)

    # print("----------------------------------")
    # print(len(generation_output.hidden_states[0]))
    # rl = []
    # for itm in generation_output.hidden_states[0]:
    #     rl.append(len(itm))
    # print(rl)

    # if args.hidden_states is not None:
    #     # print("here:",len(inputs),inputs['input_ids'].shape)
    #     if debug: 
    #         print("debug => shape:",len(generation_output.hidden_states[0]))
            

    #     for item in generation_output.hidden_states[0]:#every ids/layer?
    #         # print(item.shape)

        #     if debug:
        #         debug_tmp = []
        #         for idx,states in enumerate(item[0]):
        #             print(idx)
        #             tmp_state = states.to('cpu')
        #             tmp_state = tmp_state.reshape(1, 4096)
        #             debug_tmp.append(pd.DataFrame(tmp_state.numpy()))
        #         debug_tmp = np.array(debug_tmp)
        #         results_hs.append(debug_tmp)
            
        # for i,rei in enumerate(results_hs):
        #     if i == 0:
        #         st = pd.DataFrame(rei)
        #         st.to_csv(f"./result/debug/embedding.csv",index=False)
        #     else:
        #         st = pd.DataFrame(rei.reshape(10,4096))
        #         st.to_csv(f"./result/debug/layer_{i}.csv",index=False)
        # print("debug over")
        # exit()   


########################################################## debug
        #     last = item[0][-1]
        #     last = last.to("cpu")
        #     last = last.reshape(1, 4096)
        #     tmp.append(pd.DataFrame(last.numpy()))
        # tmp = np.array(tmp)
        # tp = pd.DataFrame(tmp.reshape(33,4096))
        
        # results_hs.append(tmp)

    # s = generation_output.sequences[0]
    # output = tokenizer.decode(s,skip_special_tokens=True)
    # response = output
    # results.append(response)
    # print(index)
    # print(response[len(dialogs[index]):])

print("=================================================")

# print(results)
# with open('./alpaca_eval/llama_base.jsonl', 'w', encoding='utf-8') as fw:
#     fw.writelines(json.dumps(item) +'\n' for item in results)

# results_tag = []
# for dia,seq in zip(dialogs,results):
#     results_tag.append(seq[len(dia) - 3:])
print("===========finish=========")
# with open('./tmp_2.log', 'w', encoding='utf-8') as fw:
    # 将每个字典转换为JSON格式的字符串，并用换行符分隔
    # fw.writelines(json.dumps({"result":item}, ensure_ascii=False) +'\n' for item in results_tag)
    # i = 0
    # j = 0
    # for idx,item in enumerate(results_tag):
    #     tp = pd.DataFrame(results_hs[idx].reshape(33,4096))
    #     # fw.write(json.dumps({"result":item}, ensure_ascii=False) +'\n')
    #     # if (find_first_digit(item) == find_first_digit(tag)):
    #     tp.to_csv(f"./hidden_state/safe/result_{i}.csv",index=False)
    #     i += 1
        
        
        # else:
        #     tp.to_csv(f"./result/llama_3_8B/neg/result_{j}.csv",index=False)
        #     j += 1





# with open('./result/toxC_500_plus.jsonl', 'w', encoding='utf-8') as fw:
#     i = 0
#     j = 0
#     for idx,(item,tag) in enumerate(zip(results_tag,tag_list)):
#         tp = pd.DataFrame(results_hs[idx].reshape(33,4096))
#         fw.write(json.dumps({"result":item,"tag":tag}, ensure_ascii=False) +'\n')
#         # if (find_first_digit(item) == tag):
#         #     tp.to_csv(f"./result/pos/result_{i}.csv",index=False)
#         #     i += 1
#         # else:
#         #     tp.to_csv(f"./result/neg/result_{j}.csv",index=False)
#         #     j += 1

