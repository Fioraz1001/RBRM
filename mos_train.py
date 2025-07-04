from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
import json
import os
import argparse
import csv
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import transformers
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F

print(torch.__version__)

print(torch.cuda.is_available())



class SVM(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)  
        
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

if torch.cuda.is_available():
    device = torch.device(0)
    print("here:",device)


def find_first_digit(string):
    match = re.search(r'[12]', string)
    if match:
        return match.group()
    else:
        return None

parser =  argparse.ArgumentParser()

parser.add_argument('--data_path',type=str)
parser.add_argument('--model_path',type=str)
parser.add_argument('--temperature',default=1,type=float)
parser.add_argument('--length',default=1000,type=int)
parser.add_argument('--test_idx',default=0,type=int)
parser.add_argument('--write_path')
parser.add_argument('--dim',default=4096,type=int)
parser.add_argument('--layer',default=40,type=int)
parser.add_argument('--model_name',type=str)

debug = False
args = parser.parse_args()

file_path = args.data_path
model_dir = args.model_path
temperature = args.temperature

length = args.length
test_idx = args.test_idx

write_path = args.write_path

dim = args.dim
layer = args.layer

model_name = args.model_name

best_mos = None
best_acc = 0
best_acc_soft = 0
best_acc_true = 0

# print("data_path:",file_path)
# print("model_path:",model_dir)
# print(os.environ["CUDA_VISIBLE_DEVICES"])

generation_config = dict(
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.1,
    output_hidden_states=True,
    return_dict_in_generate=True,
    max_new_tokens=1
    )

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


model = LlamaForCausalLM.from_pretrained (model_dir,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained (model_dir)
tokenizer.pad_token = tokenizer.eos_token


print(model)
# exit()

if dim == 4096:
    mos_model = torch.load('./mos/mos_soft_vote.pth').to(model.device)
else:
    svms_list = list(range(layer))
    mos_model = MoS_alter_fx(svms_list,dim,model.device).to(model.device)
criterion = nn.BCEWithLogitsLoss()



optimizer = optim.Adam(mos_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)


print("==============================================")
max_batch_size = 128


dialogs = [] 
tag_list = []
inst_len = len("Please respond to my input actively and cooperatively." + "input:")
fp = open(file_path,'r',encoding='utf-8')
idx = 0
cnt = 0

for line in fp.readlines():
    if cnt < test_idx:
        cnt += 1
        continue
    dt = json.loads(line)
    flag = dt['safer_response_id']
    if flag == 0:
        flag_r = 1
    else:
        flag_r = 0
    content_0 = "Please respond to my input actively and cooperatively." + "input:" + dt['prompt'] + "output:" + dt['response_{}'.format(flag)] + '</s>'
    # content_0 = dt['input'] + "Assistant:"
    content_1 = "Please respond to my input actively and cooperatively." + "input:" + dt['prompt'] + "output:" + dt['response_{}'.format(flag_r)] + '</s>'
    dialogs.append({'chosen':content_0,"rejected":content_1})

    cnt += 1
    if cnt >= test_idx + length:
        break

datasize = len(dialogs)
print(datasize)
print(model.device)
print("Start inferencing.")
results = []
results_hs = []

data_chosen = []
data_rejected = []

for index, example in tqdm(enumerate(dialogs)):
    tmp_c = []
    tmp_r = []
    input_text = [example['chosen'], example['rejected']]
    inputs = tokenizer(input_text,return_tensors="pt",padding=True,truncation=True)  #add_special_tokens=False ?
    sample_output = model.generate(
        input_ids = inputs["input_ids"].to(device),
        attention_mask = inputs['attention_mask'].to(device),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **generation_config
    )
    sample_result = []
    layers_hs = []

    for item in sample_output.hidden_states[0]:
        # print(len(item))
        # print(item.size())
        last_c = item[0][-1]
        last_r = item[1][-1]
        avg_c = item[0][0]
        avg_r = item[1][0]
        l_c = len(item[0])
        l_r = len(item[1])
        for ts in item[0][1:]:
            avg_c += ts
        for ts in item[1][1:]:
            avg_r += ts
        avg_c = avg_c / l_c
        avg_r = avg_r / l_r

        tmp_c.append(avg_c)
        tmp_r.append(avg_r)
    if dim == 4096:
        tmp_c = torch.cat(tmp_c[1:33])
        tmp_r = torch.cat(tmp_r[1:33])
    else:
        tmp_c = torch.cat(tmp_c[1:41])
        tmp_r = torch.cat(tmp_r[1:41])

    data_chosen.append(tmp_c)
    data_rejected.append(tmp_r)

epoch = 50
split = int(len(data_chosen)* 0.9)

print("Inference end.Start training")

for e in range(epoch):
    cum_loss = 0
    soft_correct = 0
    true_correct = 0
    correct = 0
    margins = 0.5
    total = len(data_chosen[split:])
    random_list = random.sample(range(len(data_chosen[:split])), len(data_chosen[:split]))
    l = len(random_list)
    for idx in tqdm(random_list):

        optimizer.zero_grad()
        c,_ = mos_model(data_chosen[idx])
        r,_ = mos_model(data_rejected[idx])

        loss = -F.logsigmoid(c - r -1).mean() # hinge=1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cum_loss += loss.item()

        print("margin:{},c:{},r:{}".format(c - r, c,r))
        margins += c - r

    print("{},margin:{},loss:{}".format(e,margins / l,cum_loss / (2 * l)))
    

    
    for idx,dt in enumerate(data_chosen[split:]):

        c,_ = mos_model(data_chosen[idx + split])
        r,_ = mos_model(data_rejected[idx + split])
        pd_c = torch.sign(c)
        pd_r = torch.sign(r)
        if pd_c >= 0: correct+=1
        if pd_r <= 0: correct+=1
        if c > r: soft_correct+=1
        if pd_c > 0 and pd_r < 0: true_correct += 1
    print("epoch:{},eval_acc:{},soft_acc:{}, truth_acc:{}".format(e,correct / (2 * total),soft_correct / total, true_correct / total))

    save_name = f'./hybrid-rm_{model_name}'

    if correct > best_acc:
        best_acc = correct
        torch.save(mos_model, save_name + 'hard' + '.pth')
    
    if soft_correct > best_acc_soft:
        best_acc_soft = soft_correct
        torch.save(mos_model, save_name + 'soft' + '.pth')

    if true_correct > best_acc_true:
        best_acc_true = true_correct
        torch.save(mos_model, save_name + 'true' + '.pth')

print("Training end.")
print("===========finish=========")
