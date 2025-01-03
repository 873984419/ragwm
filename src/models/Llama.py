import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import sys
from huggingface_hub import login
 
sys.path.append('/data/sunmengjie/lpz/ragwm')

from src.utils import load_json
from src.models.Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        
        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        
        # Hugging Face login
        login(token=hf_token)
        
        # Load tokenizer and model from the Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16).to(self.device)
        
        # # If multiple GPUs are available, use DataParallel for multi-GPU support
        # print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=config["params"]["device_ids"])
            
        # # Move model to GPU
        # self.model = self.model.to(self.device)

    def query(self, msg):
        # Tokenize input and move tensors to the proper device
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        
        # Generate model output
        outputs = self.model.generate(input_ids,
                                      temperature=self.temperature,
                                      max_new_tokens=self.max_output_tokens,
                                      early_stopping=True)
        # Decode output tokens
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return the generated response after the prompt
        return out[len(msg):]


    def calculate_perplexity(self, text):
        # 将输入文本tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # 如果有GPU，将输入也移到GPU
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        # 关闭梯度计算（不需要反向传播）
        with torch.no_grad():
            # 获取模型输出的logits
            outputs = self.model(input_ids)
            logits = outputs.logits

        # 移位以得到每个token的下一个token的logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 计算每个token的困惑度
        perplexity = torch.exp(loss.mean()).item()

        return perplexity


def parse_args():
    parser = argparse.ArgumentParser(description='Test Llama model query')
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--gpu_id', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    model_config = '/data/sunmengjie/lpz/ragwm/model_configs/llama7b_config.json'
    config = load_json(model_config )
    llm = Llama(config)
    # Query the model with a prompt
    query_prompt = "为什么 from langchain_experimental.graph_transformers import LLMGraphTransformer 提取节点为空"
    response = llm.query(query_prompt)

    # Print the model's response
    print(response)

    # 示例文本
    text = "This is a test sentence to calculate perplexity."
    
    # 计算困惑度
    ppl = llm.calculate_perplexity(text)
    print(f"Perplexity: {ppl}")