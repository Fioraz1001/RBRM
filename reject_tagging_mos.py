from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
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
parser.add_argument('--write_path',type=str)
parser.add_argument('--hidden_states',default='Yes')
parser.add_argument('--dim',default=4096,type=int)





debug = False


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# print(os.environ["CUDA_VISIBLE_DEVICES"],os.environ["WORLD_SIZE"])

file_path = args.data_path
model_dir = args.model_path
adapter_path = args.lora_path
mos_path = args.mos_path
write_path = args.write_path
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


# generation_config = dict(
#     do_sample=True,
#     # num_beams=4,
#     top_k = 8,
#     top_p = 0.9,
#     temperature=1.4,
#     repetition_penalty=1.2,
#     output_hidden_states=False,
#     return_dict_in_generate=False,
#     num_return_sequences=8,
#     max_new_tokens=256
#     )

class SVM(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim,output_dim)  # 输入维度为2，输出维度为1
        
    def forward(self, x):
        return self.linear(x)

class MoS(nn.Module):
    def __init__(self,output_dim,svms_path,svm_list,device):
        super(MoS, self).__init__()
        self.device = device
        self.svms = []  
        for i in svm_list:
            self.svms.append(torch.load(svms_path.format(i)).to(self.device))
        for svm in self.svms:
            for param in svm.parameters():
                param.requires_grad = False
        # print("Mos init ---- {} --- {}".format(len(self.svms),self.svms[0]))
        self.linear = nn.Linear(len(svm_list),output_dim).to(self.device)
        

    
    def forward(self, x_list):
        predict_list = []
        for i,svm in enumerate(self.svms):
            predict_list.append(svm(x_list[i]))
        # print("here3:",len(predict_list))
        linear_input = torch.cat(predict_list, dim=0)
        # print("here4:",linear_input.size())
        return torch.sigmoid(self.linear(linear_input))


class MoS_alter_old(nn.Module):
    def __init__(self,svm_list,device):
        super(MoS_alter, self).__init__()
        self.device = device
        self.svms = []
        self.svm_num = len(svm_list)
        self.top_k = 6
        for i in svm_list:
            self.svms.append(torch.load(svms_path.format(i)).to(self.device))
            # self.svms.append(ffn(4096, 11008, 4096).to(self.device))
        
        for svm in self.svms:
            print(svm)
            print("-----------------------------------")
            for param in svm.parameters():
                param.requires_grad = True

        # print("Mos_alter init ---- {} --- {}".format(len(self.svms),self.svms[0]))

        self.gate = nn.Linear(len(svm_list) * 4096 ,len(svm_list), bias=False).to(self.device)
        self.vote = nn.Linear(len(svm_list), len(svm_list), bias=False).to(self.device)
        
        for param in self.gate.parameters():
            param.requires_grad = True
        for param in self.parameters():
            print(param)

    
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # X_mid = F.softmax(self.gate(X),dtype=torch.float)
        # print("X_mid:",X_mid)
        # X_k, select_idx = torch.topk(X_mid, self.top_k)
        # print("select_idx:",select_idx)
        # X_k /= X_k.sum()
        # print("X_k:", X_k)
        predict = 0
        predict_list = []
        tmp_list = []
        # for i,idx in enumerate(select_idx):
        #     # x_list.append(X[i * 4096 : (i + 1) * 4096])
        #     tmp = self.svms[idx](X[idx * 4096 : (idx + 1) * 4096])
        #     tmp_list.append(tmp)
        #     tmp = tmp * X_k[i]
        #     predict += tmp
        #     predict_list.append(tmp)
        
        for i,svm in enumerate(self.svms):

            tmp = self.svms[i](X[i * 4096 : (i + 1) * 4096])
            tmp_list.append(tmp)

        t = torch.cat(tmp_list, dim=0)
        vote_weight = torch.sigmoid(self.vote(t))

        for i,t in enumerate(tmp_list):
            predict += vote_weight[i] * t
        
        print("=====")
        print(predict,vote_weight,tmp_list)



        
        # for i,svm in enumerate(self.svms):

        # print("predict_list:",predict_list)
        # print("tmp_list:",tmp_list)
        # print("==========================================================")
        return predict,tmp_list

class MoS_alter(nn.Module):
    def __init__(self,svm_list,device):
        super(MoS_alter, self).__init__()
        self.device = device
        self.svms = []
        self.svm_num = len(svm_list)
        self.top_k = 6
        for i in svm_list:
            self.svms.append(torch.load(svms_path.format(i)).to(self.device))
            # self.svms.append(ffn(4096, 11008, 4096).to(self.device))
        
        for svm in self.svms:
            print(svm)
            print("-----------------------------------")
            for param in svm.parameters():
                param.requires_grad = True

        # print("Mos_alter init ---- {} --- {}".format(len(self.svms),self.svms[0]))

        self.gate = nn.Linear(len(svm_list) * 4096 ,len(svm_list), bias=False).to(self.device)
        self.vote = nn.Linear(len(svm_list), len(svm_list), bias=False).to(self.device)
        
        for param in self.gate.parameters():
            param.requires_grad = True
        for param in self.parameters():
            print(param)

    
    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predict = 0
        predict_list = []
        tmp_list = []

        for i,svm in enumerate(self.svms):

            tmp = self.svms[i](X[i * 4096 : (i + 1) * 4096])
            tmp_list.append(tmp)

        t = torch.cat(tmp_list, dim=0)
        vote_weight = torch.sigmoid(self.vote(t))

        for i,t in enumerate(tmp_list):
            predict += vote_weight[i] * t
        
        return predict,tmp_list

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
###############################

# param_list = []

# for param in pre_MoS_new.gate.parameters():
#     param_list.append(param)

# with open('./params.csv', mode='w', newline='', encoding='utf-8') as file:
#     # 创建CSV写入器
#     csv_writer = csv.writer(file)
    
#     # 遍历列表的列表，写入每一行
#     for row in param_list:
#         csv_writer.writerow(row)  # 写入一行数据


# exit()


###############################
model = LlamaForCausalLM.from_pretrained (model_dir,device_map='auto')
tokenizer = LlamaTokenizer.from_pretrained (model_dir)
tokenizer.padding_side = 'left'
# pre_MoS = torch.load("./mos_soft_vote.pth").to(device)
# pre_MoS_new = torch.load("./result/llama-2-7b/mos_full_train_pairs.pth").to(device)







if adapter_path is not None:
    model.load_adapter(adapter_path)
    model.enable_adapters()
    print("lora_path:",adapter_path)



print(model)
print("==============================================")
max_batch_size = 128


# sys_prompt = "Which response of the following statement is safer?"
dialogs = [] 
dialogs4save = []
tag_list = []

fp = open(file_path,'r',encoding='utf-8')
idx = 0
for i,line in enumerate(fp.readlines()):
    print(i)
    dt = json.loads(line)
    # if dt['safer_response_id'] == 0: unsafe_id = 1
    # else: unsafe_id = 0
    # content_0 = dt['prompt'] + dt['response_{}'.format(unsafe_id)] + '</s>'
    # content_0 = dt['prompt'] + dt['response_{}'.format(dt['safer_response_id'])] + '</s>'
    # if len(dt['output']) < 5:continue
    content_0 = dt['input'] + dt['output'] + '</s>'
    # content_0 = dt['result'] + '</s>'

    #for test#
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
if dim == 4096: 
    results_layer = [[] for _ in range(32)]
    results_layer_reject = [[] for _ in range(32)]
else:
    results_layer = [[] for _ in range(40)]

sampled_dialogs = []
rejected_dialogs = []
best_dialogs = []
rank_dialogs = []
pair_dialogs = []


print(model.device)
print("Start inference.")
results = []

tmp_score = []
if dim == 4096: 
    tmp_score_layer = [[] for _ in range(32)]
else:
    tmp_score_layer = [[] for _ in range(40)]
tmp_dt = []
tmp_dt_rej = []

# assert len(dialogs) == 32000
# exit()

for index, example in tqdm(enumerate(dialogs)):

    input_text = example
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
    tmp = []
    if debug: exit()
    tmp_list = generation_output.hidden_states


    if args.hidden_states is not None:
        # print("here:",len(inputs),inputs['input_ids'].shape)
        # if debug: 
            # print("debug => shape:",len(generation_output.hidden_states[0]))
            
        hs_ts = []
        hs_ts_mos = []
        idxx = 0
        hs_np = torch.tensor([])


        for item in generation_output.hidden_states[0]:#every ids/layer?

            last = item[0][-1]
            # last = last.to("cpu")
            # last = last.reshape(1, 4096)
            avg_c = item[0][0]
            l_c = len(item[0])
            for ts in item[0][1:]:
                avg_c += ts
            avg_c = avg_c / l_c

            # hs_ts.append(last)
            hs_ts.append(avg_c)
            # if idxx != 0:
                
            #     hs_np = torch.cat((hs_np,last.reshape(4096)),dim=0)
            #     # print("let's check:{}".format(hs_np.shape))

            #     hs_ts_mos.append(last.reshape(4096).to(device))
                

            # idxx += 1
            # tmp.append(pd.DataFrame(last.numpy()))
        
        if dim == 4096:
            hs_np = torch.cat(hs_ts[1:33])
        else:
            hs_np = torch.cat(hs_ts[1:41])

        # tmp = np.array(tmp)
        # tp = pd.DataFrame(tmp.reshape(33,4096))
        # print("debug:tp=",tp)
        
        # for idx,itp in enumerate(tp.tolist()):
        #     print(idx,"--",itp)
        # print("columns:",tp.columns)
############################################################################
        # tensor_x = torch.from_numpy(hs_np).to(device)

        #MOS OLD
        # op = pre_MoS(hs_ts_mos)

        #MOS NEW
        print("check:",hs_np.size())
        op, layer_score = pre_MoS_new(hs_np)

        # print(len(layer_score))
        # print(layer_score)
        # print(len(layer_score))
        
        # print(layer_score)
        # exit()
        for i,it in enumerate(layer_score):
            # print(i)
            tmp = it.item()
            tmp_score_layer[i].append(tmp)


        # exit()

        # penalty = 0.15
        tmp_score.append(op.item())

        dialogs4save[index]['idx'] = index % 8

        tmp_dt.append(dialogs4save[index])

        ast = dialogs4save[index]['input'] + dialogs4save[index]['output'] + '</s>'
        assert ast == example

        if index % 8 == 7:
            l = pd.Series(tmp_score).sort_values().index
            best_dialogs.append(tmp_dt[l[7]])
            for i in reversed(l):
                rank_dialogs.append(tmp_dt[i])
            for l_i in l[4:8]:
                sampled_dialogs.append(tmp_dt[l_i])
            for i,l_i in enumerate(l[:4]):
                rejected_dialogs.append(tmp_dt[l_i])

            for ly in range(len(tmp_score_layer)):
                l = pd.Series(tmp_score_layer[ly]).sort_values().index
                for l_i in l[4:8]:
                    results_layer[ly].append(tmp_dt[l_i])
                for l_i in l[0:4]:    
                    results_layer_reject[ly].append(tmp_dt[l_i])



            tmp_dt = []
            tmp_score = []
            if dim == 4096:
                tmp_score_layer = [[] for _ in range(32)]
            else:
                tmp_score_layer = [[] for _ in range(40)]


        '''
        prediction = torch.sign(op).item()
        # prediction = torch.sign(op - 0.5 - penalty).item()
        if prediction == 1:
            sampled_dialogs.append(dialogs4save[index])
        '''

        # cnt_reject = 0
        # T = 21
        # for ly,im in enumerate(hs_ts):
        #     if ly not in svm_list:continue
        #     ii = svm_list.index(ly) 
        #     # print("debug0:",im.shape)
        #     ts_dt = im.to(device)
        #     # print("debug:",ts_dt.shape)
        #     # print(ts_dt)
        #     op = svms[ii](ts_dt)
        #     prediction = torch.sign(op).item()
        #     # print("prediction:",prediction)
        #     if prediction == -1:cnt_reject += 1
        # if cnt_reject < T:
        #     sampled_dialogs.append(dialogs4save[index])
        
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

with open(write_path.format('best'),'w',encoding='utf-8') as fw:
    fw.writelines(json.dumps(item) +'\n' for item in best_dialogs)

with open(write_path.format('chosen'),'w',encoding='utf-8') as fw:
    fw.writelines(json.dumps(item) +'\n' for item in sampled_dialogs)

with open(write_path.format('rejected'),'w',encoding='utf-8') as fw:
    fw.writelines(json.dumps(item) +'\n' for item in rejected_dialogs)

    
with open(write_path.format('rank'),'w',encoding='utf-8') as fw:
    fw.writelines(json.dumps(item) +'\n' for item in rank_dialogs)

# with open("./13b/rlhf.jsonl",'w',encoding='utf-8') as fw:
#     fw.writelines(json.dumps(item) +'\n' for item in rejected_dialogs)

# with open("pair_13b",'w',encoding='utf-8') as fw:
#     fw.writelines(json.dumps(item) +'\n' for item in sampled_dialogs)

# for ly in range(len(results_layer)):
#     with open("./7b/rlhf_layer{}_chosen.jsonl".format(ly),'w',encoding='utf-8') as fw:
#         fw.writelines(json.dumps(item) +'\n' for item in results_layer[ly])
#     with open("./7b/rlhf_layer{}_rejected.jsonl".format(ly),'w',encoding='utf-8') as fw:
#         fw.writelines(json.dumps(item) +'\n' for item in results_layer_reject[ly])


# with open('./result/alpaca_rm_neg.jsonl', 'w', encoding='utf-8') as fw:
#     # 将每个字典转换为JSON格式的字符串，并用换行符分隔
#     # fw.writelines(json.dumps({"result":item}, ensure_ascii=False) +'\n' for item in results_tag)
#     i = 0
#     j = 0
#     for idx,item in enumerate(results_tag):
#         tp = pd.DataFrame(results_hs[idx].reshape(33,4096))
#         fw.write(json.dumps({"result":item}, ensure_ascii=False) +'\n')
#         # if (find_first_digit(item) == find_first_digit(tag)):
#         tp.to_csv(f"./result/alpaca_s/pos/result_{i}.csv",index=False)
#         i += 1
        
        
#         # else:
#         #     tp.to_csv(f"./result/llama_3_8B/neg/result_{j}.csv",index=False)
#         #     j += 1





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

