
##############################################################################################################
##############################################################################################################

import torch
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaModel, LlamaConfig, LlamaForSequenceClassification
import os
import argparse
import torch.nn as nn
import torch.optim as optim

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
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
            tmp = self.svms[i](X[i * 5120 : (i + 1) * 5120]) ## for qwen
            tmp_list.append(tmp)
        t = torch.cat(tmp_list, dim=0)
        vote_weight = torch.sigmoid(self.vote(t))
        for i,t in enumerate(tmp_list):
            predict += vote_weight[i] * t
        return predict, tmp_list


if torch.cuda.is_available():
    device = torch.device(0)
    print("here:",device)

parser =  argparse.ArgumentParser()
parser.add_argument('--data_path',type=str)
parser.add_argument('--model_path',type=str)
parser.add_argument('--model_ref_path',type=str)
parser.add_argument('--hybrid_reward_path',type=str)
parser.add_argument('--output_dir',type=str)

args = parser.parse_args()

data_path = args.data_path
model_dir = args.model_path
model_ref_path = args.model_ref_path
output_dir = args.output_dir
hybrid_reward_path = args.hybrid_reward_path

# model_dir = ''
# data_path = ''

idx_list = [ 'response_0', 'response_1', 'is_response_0_safe', 'is_response_1_safe', 'better_response_id', 'safer_response_id']

def preprocess_function(examples):
    new_examples = {}
    # ['prompt', 'response_0', 'response_1', 'is_response_0_safe', 'is_response_1_safe', 'better_response_id', 'safer_response_id']

    flag = examples['safer_response_id']
    if flag == 1:flag_r = 0
    else:flag_r = 1
    prompt = examples['prompt']
    chosen = examples['response_{}'.format(flag)]
    rejected = examples['response_{}'.format(flag_r)]

    new_examples['prompt'] = prompt
    new_examples['chosen'] = chosen
    new_examples['rejected'] = rejected

    return new_examples



dataset = load_dataset("json", data_files=data_path)

print(dataset)
max_length = 4096

dataset = dataset.filter(
    lambda x: len(x["chosen"]) <= max_length
    and len(x["rejected"]) <= max_length
)

model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float32,device_map='auto')
model_ref = model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float32,device_map='auto') 

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token


dpo_args = DPOConfig(

    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_length=max_length,

    warmup_ratio=0.03,
    num_train_epochs=1,
    learning_rate=1e-6,
    bf16=True,
    logging_steps=1,
    optim="adamw_torch",
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=100,
    max_steps=100,
    output_dir=output_dir,
    save_total_limit=1,
    

    
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=dpo_args,
    beta=0.15,
    train_dataset=dataset['train'],
    tokenizer=tokenizer,
    model_init_kwargs=None,
    hybrid_reward_path=hybrid_reward_path
)


dpo_trainer.train()
dpo_trainer.save_model()


