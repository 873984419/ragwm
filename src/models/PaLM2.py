import google.generativeai as palm
import google.ai.generativelanguage as gen_lang
import time
import torch
import argparse
import sys
sys.path.append('/data/sunmengjie/lpz/ragwm')

from src.utils import load_json
from src.models.Model import Model
# from .Model import Model

 
 

class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
         
        api_pos = int(config["api_key_info"]["api_key_use"])
        api_key = config["api_key_info"]["api_keys"][api_pos]
        palm.configure(api_key=api_key )
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.model = palm.GenerativeModel(model_name='gemini-1.5-flash')
            
    def query(self, msg):

        try:
            # print(f'palm msg: {msg}')
            response = self.model.generate_content(msg ).text

        except Exception as e:
            print(e)
            if 'not supported' in str(e):
                return ''
            elif 'Unknown field for Candidate' in str(e):
                response = 'Input may contain harmful content and was blocked by PaLM.'
            else:
                print('Error occurs! Please check output carefully.')
                time.sleep(300)
                return self.query(msg)
 

        return response
    
def parse_args():
    parser = argparse.ArgumentParser(description='Test Llama model query')
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    model_config = '/data/sunmengjie/lpz/ragwm/model_configs/palm2_config.json'
    config = load_json(model_config )
    llm = PaLM2(config)
    # Query the model with a prompt
    query_prompt = '1+9?'
    response = llm.query(query_prompt)

    # Print the model's response
    print(response)


### note proxy in amrican