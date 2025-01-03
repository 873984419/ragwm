import sys, os
from contriever_src.contriever import Contriever
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
import hashlib
from collections import defaultdict
import random
import torch
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer
import re
# 用于指定查询模型（qmodel）和候选模型（cmodel）
model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

# 不同的模型可能有不同的输出方式，因此定义了不同的获取嵌入函数
def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

# https://github.com/facebookresearch/contriever
def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
 
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb

def load_beir_datasets(dataset_name, split):
    # https://huggingface.co/datasets/BeIR/nq
    assert dataset_name in ['nq','msmarco','hotpotqa','nfcorpus','trec-covid']
    if dataset_name == 'msmarco': split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    # out_dir = os.path.join(os.getcwd(), "datasets")
    out_dir = "/data/sunmengjie/lpz/ragwm/datasets"
    data_path = os.path.join(out_dir, dataset_name)
    print(data_path)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)    

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    # 获取文件夹路径
    print(file_path)
    dir_path = os.path.dirname(file_path)
    # 如果文件夹不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f, indent=4)
def create_file_if_not_exists(file_path):

    # 检查文件是否存在

    if not os.path.exists(file_path):

        # 如果文件不存在，则创建文件

        with open(file_path, 'w') as file:

            file.write('')  # 可选：写入一些初始内容

        print(f"文件 '{file_path}' 已创建。")

    else:

        print(f"文件 '{file_path}' 已存在。")
        
def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores



def extract_doc(WT):
    ### Return only the refined watermark text in the following JSON format: 
    # [{{"watermark_text": "Your generated text 1"}}]
    # pattern = re.compile(r'\[.*?\]', re.DOTALL) ### ```json [....] ```
    # r'{.*}'
    
    pattern = re.compile(r'{.*}', re.DOTALL)
    matches = pattern.findall(WT)
    # logger.info(f'matches: {matches[0]}', is_valid_json(matches[0]))
    if len(matches) and is_valid_json(matches[0]):
        # Parse the JSON data
        watermarks = json.loads(matches[0])
        print(f'watermarks: {watermarks}, {type(watermarks)}')
        print( watermarks['watermark_text'])

        # Extract the texts
        # watermark_text_list = [entry["watermark_text"] for entry in watermarks]
        # watermark[0]['watermarktext']
    return  watermarks['watermark_text']

def extract_doc_list(WT):
    # Return exactly 5 distinct sentences, formatted as a JSON list, like this:
    # [
    #     "Sentence 1.",
    #     "Sentence 2.",
    #     "Sentence 3.",
    #     "Sentence 4.",
    #     "Sentence 5."
    # ]
    pattern = re.compile(r'\[.*?\]', re.DOTALL) ### ```json [....] ```
    matches = pattern.findall(WT)
    if len(matches) and is_valid_json(matches[0]):
        # Parse the JSON data
        WT_list = json.loads(matches[0])
    print(WT_list)
    return WT_list

def is_valid_json(data: str) -> bool:
    """ check whether the data is in json format """

    try:
        json.loads(data)
    except ValueError:
        return False
    return True




def find_substrings_containing_sentences(text, sentences):
    def split_text(text):
        # 使用正则表达式按句号或换行符划分文本
        return re.split(r'\.|\n', text)

    def contains_sentences(substring, sentences):
        # 检查子字符串是否包含任意一个句子，不区分大小写
        return all(sentence.lower() in substring.lower() for sentence in sentences)

    def filter_substrings(text, sentences):
        # 划分文本
        substrings = split_text(text)
        # 筛选包含指定句子的子字符串
        matched_substrings = [substring for substring in substrings if contains_sentences(substring, sentences)]
        return matched_substrings
    
    # 筛选符合条件的子字符串
    matched_substrings = filter_substrings(text, sentences)
    
    # 去掉每个子字符串的前后空白字符，并拼接成一段话
    concatenated_text = '. '.join(substring.strip() for substring in matched_substrings if substring.strip())


    
    return [concatenated_text+'.', len(matched_substrings)]

import logging


class Log:
    """
    wrapper of logging module
    """

    def __init__(self, log_file , file_level=logging.DEBUG, console_level=logging.INFO) -> None:
        self.log_file = log_file
        self.file_level = file_level
        self.console_level = console_level

    def get(self, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.log_file, 'a')
        file_handler.setLevel(self.file_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)

        date_fmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt=date_fmt)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger


def file_exist(file_path):
        # 获取文件夹路径
    print(file_path)
    dir_path = os.path.dirname(file_path)
    # 如果文件夹不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        # 判断文件是否存在
    if not os.path.exists(file_path):
        # 文件不存在，创建文件
        with open(file_path, 'w') as file:
            file.write("This is a newly created file.\n")
    else:
        print(f"The file '{file_path}' already exists.")

# wh:将文本hash
def documents_hash(documents):
    hash_documents = []
    for index,document in enumerate(documents):
        sha256_obj = hashlib.sha256()
        sha256_obj.update((str(document)).encode('utf-8'))
        sha256_hash = sha256_obj.hexdigest()
        hash_documents.append(sha256_hash)
    return hash_documents

# wh:返回要删除的documents的id列表 
def remove_duplicates_with_indices(results):
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]


    documents = documents_hash(results["documents"][0])

    seen = set()  # 用于追踪已经出现的元素
    indices_removed = []  # 存储被删除元素的索引
    result = []
    delete_ids = []  # 存储要删除元素的ids
    
    for i, value in enumerate(documents):
        if value not in seen:
            seen.add(value)
            result.append(value)
        else:
            indices_removed.append(i)  # 记录重复元素的索引
            if(metadatas[i]["change"]):
                delete_ids.append(ids[i])
    
    return delete_ids, indices_removed