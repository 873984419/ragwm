import json
import hashlib
import os
import random
from collections import Counter
import argparse
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)  # Add project root to sys.path

from src.utils import load_json, save_json

### keep hash value in 'hash.txt', entity hash list in 'entity_hash.json', and relation hash list
 
 
class EntityProcessor:
    def __init__(self):
        self.entity_types = {}
        self.entities = {}
        self.relation_types = {}

    def get_top_entity_types(self, k=10):
        """Get top k entity types by frequency."""
        # entity_types = sorted(self.entity_types, key=lambda x: x[1], reverse=True)[:k]
        entity_types = sorted(self.entity_types.items(), key=lambda item: item[1], reverse=True)
        return [item[0] for item in entity_types[:k]]
    
    def get_top_relation_types(self, k=10):
        """Get top k relation types by frequency."""
         
        relation_types = sorted(self.relation_types.items(), key=lambda item: item[1], reverse=True) 
        return [item[0] for item in relation_types[:k]]


    def filter_entities(self, entity_types):
        """Filter entities based on the top entity types."""
        return [eid for eid, info in self.entities.items() if info in entity_types]
    
    def get_hash_init(self):
        """Generate an initial random hash."""
        random_bytes = os.urandom(32)
        random_hex = random_bytes.hex()
        print(random_bytes, random_hex)
        return random_hex  

    def hash_entity(self, filtered_entities , hash_init, hash_num=10):
        """Generate hash values for entities."""
        # hash init and hash entity and save as 'entity1, entity2, entity3, ...'
        entity_total = len(filtered_entities)
        entity_hash_list = []

        sha256_hash = hashlib.sha256(hash_init.encode('utf-8')).hexdigest()
        init_hash = sha256_hash
        hash_entity = filtered_entities[int(sha256_hash,16) % entity_total]
        
        entity_hash_list.append(hash_entity)
        
        for i in range(hash_num-1):

            sha256_obj = hashlib.sha256()
            # 更新哈希对象以使用字符串\n",
            sha256_obj.update((entity_hash_list[i]+str(init_hash)).encode('utf-8'))
            # 获取SHA-256哈希值\n",
            sha256_hash = sha256_obj.hexdigest()
            hash_entity = filtered_entities[int(sha256_hash,16) % entity_total]
            entity_hash_list.append(hash_entity)
        
        return entity_hash_list
        # hash_file_path = 'hash.txt'


def hash_wmunit(entity_hash_list, relation_type_list, hash_init ):
    """Generate watermark units based on hashed entities and relations."""
    relation_total = len(relation_type_list)

    def get_wmunit(s1, t1 ):

        sha256_obj = hashlib.sha256()
        # 更新哈希对象以使用字符串\n",
        sha256_obj.update((s1+t1+str(hash_init)).encode('utf-8'))
        # 获取SHA-256哈希值\n",
        sha256_hash = sha256_obj.hexdigest()

        return relation_type_list[int(sha256_hash, 16) % relation_total]

    wmunit_list = []
    entity_total = len(entity_hash_list)
    count = 0
    for i in range(entity_total):
        for j in range(i, entity_total):
            print(i,j)
            # continue
            if i == j:
                continue
            # elif j - i == 1:
            #     wmunit_list.append((entity_hash_list[i], entity_hash_list[j], get_wmunit(entity_hash_list[i], entity_hash_list[j])))
            elif random.random() < args.edge_prob:
                wmunit_list.append((entity_hash_list[i], entity_hash_list[j], get_wmunit(entity_hash_list[i], entity_hash_list[j])))
            count +=  1
        
    print(f"Total watermark units: {len(wmunit_list)}, count:{count}")
    return wmunit_list


 
 
def scratch_run():
    """Initialize and process entities, relations, and watermarks."""
    entityprocessor = EntityProcessor()
    entityprocessor.entities = load_json(entities_dict_path)
    entityprocessor.entity_types = load_json(entity_type_path)
    entityprocessor.relation_types = load_json(relation_type_path)

    entity_top = entityprocessor.get_top_entity_types(k=20)
    # save_json(entity_top, entity_top_path)

    filter_entity = entityprocessor.filter_entities(entity_top)
    print('filter_entity', len(filter_entity))
    save_json(filter_entity, entity_top_list_path)

    exit()
    hash_init = entityprocessor.get_hash_init()
    print(hash_init)
    # save_json(hash_init, init_hash_path)

    entity_hash = entityprocessor.hash_entity(filter_entity, hash_init, hash_num=args.entity_num)
    # save_json(entity_hash, entity_hash_path)

    relation_top = entityprocessor.get_top_relation_types(k=20)
    # save_json(relation_top, relation_top_path)

    wmunit_list = hash_wmunit(entity_hash, relation_top,hash_init  )
    # save_json(wmunit_list, watermark_unit_path)


def wmunit_run():
    """Process watermark units based on existing hashes and relations."""
    entity_hash = load_json( entity_hash_path)
    relation_top = load_json(relation_top_path)
    hash_init = load_json(init_hash_path)
    print(hash_init)
    wmunit_list = hash_wmunit(entity_hash, relation_top, hash_init)
    save_json(wmunit_list, watermark_unit_path)


def check_run():
    """Check if watermark units exist in the relation list."""
    relation_list = load_json(relation_list_path)
    wmunit_list = load_json(watermark_unit_path)
    # print(relation_list)
    for wmunit in wmunit_list:
 
        if tuple(wmunit) in relation_list:
            print(f"{wmunit} 存在于关系列表中")
            print(f"{wmunit} exists in the relation list")
    


def cal_run():
    """Count the number of entities and relationships"""
    relation_list = load_json(relation_list_path)
    entity_list = load_json(entities_dict_path)
    print(len(relation_list) , len(entity_list))
    
    

def parse_args():
    parser = argparse.ArgumentParser(description='Generate watermark unit list')
    parser.add_argument('-t', choices=['scratch', 'wmunit', 'check', 'cal'], default='wmunit', help='Run type')

    # BEIR datasets
        
    parser.add_argument('--eval_dataset', type=str, default='trec-covid', help='BEIR dataset to evaluate', choices= ['trec-covid','nfcorpus','nq', 'msmarco', 'hotpotqa'])
     #['nq','msmarco','hotpotqa','nfcorpus','trec-covid']
    parser.add_argument('--split', type=str, default='test')

    #save path 
    parser.add_argument('--basepath', type=str, default='/data/sunmengjie/lpz/ragwm/output/wm_prepare')

    ## hash para
    parser.add_argument('--entity_num', type=int, default=100)
    parser.add_argument('--edge_prob', type=float, default=0.05, help=' 100*98/2*0.05 + 100 = 345')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    basepath =  args.basepath 
 

    # Define paths for saving the outputs
    entities_dict_path = os.path.join(basepath, args.eval_dataset, 'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath,  args.eval_dataset, 'entity_type_llm.json')
    relation_type_path = os.path.join(basepath, args.eval_dataset,  'relation_type_llm.json')
    relation_list_path = os.path.join(basepath, args.eval_dataset, 'relation_list_llm.json')


    entity_top_path = os.path.join(basepath, args.eval_dataset, 'entity_top.json') 
    entity_top_list_path = os.path.join(basepath, args.eval_dataset, 'entity_top_list.json') 
    relation_top_path = os.path.join(basepath, args.eval_dataset, 'relation_top.json')
    init_hash_path = os.path.join(basepath, args.eval_dataset, 'init_hash.json')
    entity_hash_path = os.path.join(basepath, args.eval_dataset, 'entity_hash.json')
    watermark_unit_path = os.path.join(basepath, args.eval_dataset, 'wmunit.json')

 
    if args.t == 'scratch':
        scratch_run()
    elif args.t == 'wmunit':
        wmunit_run()
    elif args.t == 'check':
        check_run()
    elif args.t == 'cal':
        cal_run()

### note
# 

### smj 
# python generate_hash_entity.py --eval_dataset "trec-covid" -t scratch --entity_num 100 --edge_prob 0.05
# python generate_hash_entity.py -t check
# python generate_hash_entity.py --eval_dataset "nfcorpus" -t scratch --entity_num 100 --edge_prob 0.05
#  python generate_hash_entity.py --eval_dataset 'hotpotqa' -t scratch --entity_num 100 --edge_prob 0.05

#python generate_hash_entity.py --eval_dataset 'msmarco' -t scratch --entity_num 100 --edge_prob 0.05