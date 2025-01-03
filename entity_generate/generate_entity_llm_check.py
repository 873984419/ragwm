import json
import os
import time
import pickle
import traceback
from datasets import load_dataset

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

import sys
import argparse
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(project_root)
sys.path.append(project_root)
exit()
from src.utils import load_beir_datasets, load_json, save_json

from src.utils import Log, file_exist
def data_prepare(dataname, dataset):
    ndataset = {}
    if dataname in ['nq', 'msmarco', 'hotpotqa', 'nfcorpus', 'trec-covid']:
        for key, value in dataset.items():
            ndataset[key] = value['text']
    elif dataname == 'closed_qa':
        for i, item in enumerate(dataset):
            ndataset[str(i)] = f"{item['instruction']}. {item['context']}"
    return ndataset


def generate_entity(dataset, basepath, prob=0.01):
    entities_dict = {}
    node_type_dict = {}
    relation_type_dict = {}
    unique_relation = []
    current_key = ''
    tcount =0
    ## load has saved results
    try:
        entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key = load_checkpoint(basepath)
        print(current_key,type(current_key))
        print("successfully load saved file")
    except Exception as e:
        print(f"Error occurred while converting document to graph: {e}")
        traceback.print_exc()
             
    # 按 key 排序遍历
    for i, key in enumerate(sorted(dataset.keys())):
        # logger.info(f'i: {i}, key:{key}')
        if current_key!= '' and key < current_key:
            # 跳过已处理的 key
            # print(f'key has run, key:{key}, current_key:{current_key}')
            continue

        if random.random() > prob:
            # print(f'jump within prob: {prob}')
            continue    
        item = dataset[key]
        print(f"Processing item {i}: key = {key},tcount: {tcount} ")
        # print(f'item: {item}')
        tcount+=1
        # try:
        documents = [Document(page_content=item)]
        print(f'documents:{documents}')
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        nodes_list = graph_documents[0].nodes
        # except Exception as e:
        #     print(f"Error occurred while converting document to graph: {e}")
        #     traceback.print_exc()
        #     continue

        # 处理 nodes
        # print(f'node_list:{nodes_list}')
        for node in nodes_list:
            node_type_dict[node.type] = node_type_dict.get(node.type, 0) + 1
            if node.id not in entities_dict:
                entities_dict[node.id] = node.type

        relations_list = graph_documents[0].relationships

        # 处理 relations
        for relation in relations_list:
            relation_type_dict[relation.type] = relation_type_dict.get(relation.type, 0) + 1
            if [relation.source.id, relation.target.id, relation.type] not in unique_relation:
                unique_relation.append([relation.source.id, relation.target.id, relation.type])

        if tcount % 20  == 1 :
            print(i, time.time())
            
            # 每次处理完当前 key，存储当前中间结果
            save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation , key, basepath)
            # break
    save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation,   key, basepath)
    print(f'all dataset has been processed')
 

    return True


def save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key, basepath):


    # Define paths for saving the outputs
    entities_dict_path = os.path.join(basepath,   'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath,   'entity_type_llm.json')
    relation_type_path = os.path.join(basepath,  'relation_type_llm.json')
    relation_list_path = os.path.join(basepath,  'relation_list_llm.json')

    checkpoint_file = os.path.join(basepath,  'checkpoint.json')
    save_json( current_key, checkpoint_file)

    save_json(entities_dict, entities_dict_path)
    save_json(node_type_dict, entity_type_path)
    save_json(relation_type_dict, relation_type_path)
    save_json( unique_relation, relation_list_path)

def load_checkpoint(  basepath):
 
    # Define paths for saving the outputs
    entities_dict_path = os.path.join(basepath,   'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath,   'entity_type_llm.json')
    relation_type_path = os.path.join(basepath,  'relation_type_llm.json')
    relation_list_path = os.path.join(basepath,  'relation_list_llm.json')
    checkpoint_file = os.path.join(basepath,  'checkpoint.json')


    entities_dict = load_json(entities_dict_path)
    node_type_dict = load_json(entity_type_path)
    relation_type_dict = load_json(relation_type_path)
    unique_relation = load_json(relation_list_path)
    current_key =load_json(checkpoint_file)
    
    return entities_dict, node_type_dict, relation_type_dict, unique_relation,  current_key


def parse_args():
    parser = argparse.ArgumentParser(description='Generate entities and relations from dataset')
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    # parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default='hotpotqa', help='BEIR dataset to evaluate', choices= ['trec-covid','nfcorpus','nq', 'msmarco', 'hotpotqa'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--basepath', type=str, default='/home/sunmengjie/lpz/vectordb/ragwatermark/output/wm_prepare')
    parser.add_argument('--dataset_prob', type=float, default=0.01, help='for nfcorpus 1')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # LLM setup
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    config = load_json(args.model_config_path)

    api_keys = config["api_key_info"]["api_keys"][0]
    api_base = config["api_key_info"]["api_base"][0]
    os.environ["OPENAI_API_KEY"] = api_keys
    os.environ["OPENAI_API_BASE"] = api_base

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_transformer = LLMGraphTransformer(llm=llm)

    # Load dataset
    if args.eval_dataset == 'closed_qa':
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        ndataset = data_prepare('closed_qa', closed_qa_dataset)
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        ndataset = data_prepare(args.eval_dataset, corpus)

    basepath = os.path.join(args.basepath, args.eval_dataset)
    LOG_FILE = os.path.join(basepath,'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    generate_entity(ndataset, basepath, prob=args.dataset_prob)

 
### smj
# 'nfcorpus 3633 item, prob is set 1
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'nfcorpus' --dataset_prob 1
# 'trec-covid 171332 item, prob is set 0.05 , 
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'trec-covid' --dataset_prob 0.03
# total 2681468  prob is set 0.002
# python entity_generate/generate_entity_llm_check.py  --eval_dataset 'nq' --dataset_prob 0.0019
# total 8841823  prob is set 0.001
# python entity_generate/generate_entity_llm_check.py --eval_dataset 'msmarco' --dataset_prob 0.00057
# total 5233329 prob is set 0.001
# python entity_generate/generate_entity_llm_check.py  --eval_dataset  'hotpotqa'  --dataset_prob 0.00096