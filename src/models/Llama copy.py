import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
        # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
# https://huggingface.co/docs/transformers/main/en/model_doc/llama#usage-tips
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(os.path.dirname(__file__))
# 添加项目根目录到sys.path
sys.path.append(project_root)
from src.utils import load_json
from  src.models.Model import Model

# from .Model import Model

from huggingface_hub import login

class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        print(self.name)
        login(token=hf_token)
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        # self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # self.model = LlamaForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16, use_auth_token=hf_token).to(self.device)



        self.tokenizer = AutoTokenizer.from_pretrained(self.name )
        self.model = AutoModelForCausalLM.from_pretrained(self.name) 
        print(torch.cuda.device_count())
        
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.to('cuda:0') 
    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_idsto('cuda:0') 
        outputs = self.model.generate(input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True)
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = out[len(msg):]
        return result
    


def parse_args():
    parser = argparse.ArgumentParser(description='test')
        # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='llama7b')
    args = parser.parse_args()
    print(args)
    return args
if __name__ == '__main__':
    args = parse_args()
    print(torch.cuda.device_count())
    device_ids = [0, 1, 2, 3]
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    config = load_json(args.model_config_path)
    print(args.model_config_path )
    print(config)
    llm = Llama(config )
    query_prompt = "为什么 from langchain_experimental.graph_transformers import LLMGraphTransformer 提取节点为空"
    response = llm.query(query_prompt)
    print(response)
