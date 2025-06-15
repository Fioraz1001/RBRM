from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import json
import os
import argparse
import csv
import pandas as pd
import numpy as np
import re
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
parser.add_argument('--mos_path',type=str)
parser.add_argument('--accept_path',type=str)
parser.add_argument('--greedy_path',default=None,type=str)
parser.add_argument('--hidden_states',default=True)
parser.add_argument('--dim',default=4096,type=int)





debug = False


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# print(os.environ["CUDA_VISIBLE_DEVICES"],os.environ["WORLD_SIZE"])

file_path = args.data_path
model_dir = args.model_path
adapter_path = args.lora_path
mos_path = args.mos_path
write_path = args.accept_path
greedy_path = args.greedy_path
dim = args.dim

print("data_path:",file_path)
print("model_path:",model_dir)

generation_config = dict(
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.1,
    output_hidden_states=True,
    return_dict_in_generate=True,
    max_new_tokens=1,
    )


class SVM(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)  # 输入维度为2，输出维度为1
        
    def forward(self, x):
        return self.linear(x)

class MoS_alter_fx(nn.Module):
    def __init__(self,svm_list,dim,device):
        super(MoS_alter_fx, self).__init__()
        self.device = device
        self.svms = []
        self.svm_num = len(svm_list)
        for i in svm_list:
            self.svms.append(SVM(dim,1).to(self.device))
            # self.svms.append(ffn(4096, 11008, 4096).to(self.device))
        self.gate = nn.Linear(len(svm_list) * dim ,len(svm_list), bias=False).to(self.device)
        self.vote = nn.Linear(len(svm_list), len(svm_list), bias=False).to(self.device)
        

    
    def forward(self, X):
        # X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predict = 0
        tmp_list = []
        for i,svm in enumerate(self.svms):
            tmp = self.svms[i](X[i * dim : (i + 1) * dim])
            tmp_list.append(tmp)
        t = torch.cat(tmp_list, dim=0)
        vote_weight = torch.sigmoid(self.vote(t))
        for i,t in enumerate(tmp_list):
            predict += vote_weight[i] * t
        return predict, tmp_list

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

TemplateLora = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
Inst = "Respond to this input."

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



pre_MoS_new = torch.load(mos_path).to(device)
print(pre_MoS_new)

model = AutoModelForCausalLM.from_pretrained (model_dir,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained (model_dir)
tokenizer.padding_side = 'left'

if adapter_path is not None:
    model.load_adapter(adapter_path)
    model.enable_adapters()
    print("lora_path:",adapter_path)

print(model)
print("==============================================")
max_batch_size = 128
dialogs = [] 
dialogs4save = []
tag_list = []

fp = open(file_path,'r',encoding='utf-8')
idx = 0
for line in fp.readlines():
    dt = json.loads(line)
    content_0 = dt['input'] + dt['output'] + '</s>'
    if debug == True:
        content_0 = "<s> Who are you? I'm Miku </s>"
        dialogs.append(content_0)
        break
    #test end#
    dialogs.append(content_0)
    dialogs4save.append(dt)

datasize = len(dialogs)
print(datasize)
results = []
results_hs = []

sampled_dialogs = []
greedy_dialogs = []
worse_dialogs = []
pair_dialogs = []
print(model.device)
print("Start inference.")
results = []

tmp_score = []

tmp_dt = []
tmp_dt_rej = []
for index, example in tqdm(enumerate(dialogs)):

    
    input_text = example
    inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
    if debug == True:
        results_hs.append(inputs["input_ids"])

    generation_output = model.generate(
        input_ids = inputs["input_ids"].to(device),
        attention_mask = inputs['attention_mask'].to(device),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        
        **generation_config
    )
    tmp = []
    if debug: exit()
    tmp_list = generation_output.hidden_states


    if args.hidden_states is not None:

        hs_ts = []
        hs_ts_mos = []
        idxx = 0
        hs_np = torch.tensor([])


        for item in generation_output.hidden_states[0]:#every ids/layer?

            last = item[0][-1]
            hs_ts.append(last)
        if dim == 4096:
            hs_np = torch.cat(hs_ts[1:33])
        else:
            hs_np = torch.cat(hs_ts[1:41])


        print("check:",hs_np.size())
        op, layer_score = pre_MoS_new(hs_np)

        tmp_score.append(op.item())
        tmp_dt.append(dialogs4save[index])
        
        if index % 8 == 7:
            l = pd.Series(tmp_score).sort_values().index
            sampled_dialogs.append(tmp_dt[l[7]])
            greedy_dialogs.append(tmp_dt[1])
            worse_dialogs.append(tmp_dt[l[0]])

            tmp_dt = []
            tmp_score = []

        results_hs.append(tmp)

    s = generation_output.sequences[0]
    output = tokenizer.decode(s,skip_special_tokens=True)
    response = output
    results.append(response)
    print(index)
    print(response[len(dialogs[index]) - 5:])


results_tag = []
for dia,seq in zip(dialogs,results):
    results_tag.append(seq[len(dia) - 3:])
print("===========finish=========")
print("sampled data num:",len(sampled_dialogs))

with open(write_path,'w',encoding='utf-8') as fw:
    fw.writelines(json.dumps(item) +'\n' for item in sampled_dialogs)

if greedy_path is not None:
    with open(greedy_path,'w',encoding='utf-8') as fw:
        fw.writelines(json.dumps(item) +'\n' for item in greedy_dialogs)

