
 

import argparse

import chromadb
 

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加项目根目录到sys.path
sys.path.append(project_root)
from src.utils import load_beir_datasets, load_models,load_json
import argparse
import torch
 
ChromadbPath = '/data/sunmengjie/chromdb_back/smj_db'
 
      
class VectorStore:
    def __init__(self, embedding_model, tokenizer, get_emb, dataset, device, collection_name, use_local, distance= 'cosine'):
 
        # self.chroma_client = chromadb.Client()
        # self.use_local = use_local
        ## will autoload and autosave update
        self.chroma_client = chromadb.PersistentClient(path=ChromadbPath ) 
        collections = self.chroma_client.list_collections()
        print(collections)
        collection_exists = any(col.name == collection_name for col in collections)
        if collection_exists and use_local:
            print(f"Here are using an exsiting local chromadb named {collection_name}!!! ")
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Here are using an exsiting local chromadb named {collection_name}!!! ")
        else:
            print('ok',collection_exists)
            if collection_exists : 
                print(f"Here are delete a existent local chromadb named {collection_name}!!!")
                # collection = self.chroma_client.get_collection(name=collection_name)
                self.chroma_client.delete_collection(name=collection_name)
            print(f"Here are creating a nonexistent local chromadb named {collection_name}!!!")
            self.collection = self.chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": distance})
        
        # Filter the dataset to only include entries with the 'closed_qa' category
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.get_emb = get_emb
        # torch.cuda.set_device(2)
        # device = 'cuda'
        self.device = device
        self.embedding_model.eval()
        self.embedding_model.to(self.device)

    def get_embedding (self, text):
            text_input = self.tokenizer(text,padding=True, truncation=True, return_tensors="pt")
            # text_input = {key: value.cuda() for key, value in text_input.items()}
            text_input = {key: value.to(self.device) for key, value in text_input.items()}

            # print(f"text_input device: {text_input['input_ids'].device}")
            # print(f"embedding_model device: {next(self.embedding_model.parameters()).device}")
            with torch.no_grad():
                text_emb = self.get_emb(self.embedding_model, text_input).squeeze().tolist()
            return text_emb

    # Method to populate the vector store with embeddings from a dataset
    def populate_vectors_backup(self):
        # {'doc0':{'text': "In accounting, minority interest (or non-controlling interest) is the portion of a subsidiary corporation's stock that is not owned by the parent corporation. The magnitude of the minority interest in the subsidiary company is generally less than 50% of outstanding shares, or the corporation would generally cease to be a subsidiary of the parent.[1]", 'title': 'Minority interest'}}
        count = 0 
        for key, value in  self.dataset.items() :
  
            # print(key, value)
            text_emb = self.get_embedding(value['text'])
            self.collection.add(embeddings=[text_emb ], documents=[value['text']], ids=[f'id_{count}'], metadatas={'title': value['title'], 'id':key , 'change':False})
            count+=1

            if count % 100 == 0:
                print(count)

    def populate_vectors(self):
            # 按 key 排序遍历
        
        current_key  =  self.collection.count()
        for count, key in enumerate(sorted(self.dataset.keys())):
            # logger.info(f'i: {i}, key:{key}')
            if count < current_key:
                # 跳过已处理的 key
                # print(f'key has run,count:{count} key:{key}, current_key:{current_key}')
                continue
                        # print(key, value)
            # print(f'key has run,count:{count} key:{key}, current_key:{current_key}')
            # print(self.dataset[key])
            value = self.dataset[key]
            text_emb = self.get_embedding(value['text'])
            self.collection.add(embeddings=[text_emb ], documents=[value['text']], ids=[f'id_{count}'], metadatas={'title': value['title'], 'id':key , 'change':False})
             

            if count % 500 == 0:
                print(count)


    # Method to search the ChromaDB collection for relevant context based on a query
    def search_context(self, query, n_results ):
        text_emb = self.get_embedding(query)
        return self.collection.query(query_embeddings=text_emb, n_results=n_results)

 
    def update_context(self,  id, str_add='', pos='end'):
        id_data = self.get_id(id)
        
        if id_data['metadatas'][0]['change'] == True and str_add=='':
            # for clean_collection
            print(id_data)
            context = self.dataset[id_data['metadatas'][0]['id']]['text']
            flag = False  
        elif id_data['metadatas'][0]['change'] == False and str_add=='':
            return True
        else:
            if pos =='end':
                context = id_data['documents'][0] + ' ' + str_add
            elif pos =='front':
                context =  str_add+ ' ' + id_data['documents'][0]
            else:
                context = id_data['documents'][0] + ' ' + str_add
            flag = True

        embeddings = self.get_embedding(context)
        metadata={'title': self.dataset[id_data['metadatas'][0]['id']]['title'], 'id':id_data['metadatas'][0]['id'], 'change':flag}

        self.collection.update(ids=[id], embeddings=[embeddings], documents =[context], metadatas=metadata)
        # self.collection.update(ids=[id], embeddings=[embeddings], documents =[context], metadatas=metadata)

    def clean_collect(self):
        print('befor clean')
        # 过滤出 metadata 中 'change' 为 True 的记录
        results = self.collection.get(
            where={'change': True} # 过滤条件
 
        )
        # print(f'results: {results}')
        if len(results['ids']) == 0 : return True
        for i in range(len(results['ids'])):
            # id_{count}
            
            doc_id = results['ids'][i]
            count = doc_id.split('_')[1]
            print('befor clean',  self.collection.count())
            if int(count) >= len(self.dataset) :
                self.collection.delete(doc_id)
                print('after delete',  self.collection.count())
            elif results['metadatas'][i]['id'] == ' ':
                self.collection.delete(doc_id)
                print('after delete',  self.collection.count())
        
        # if id_data['metadatas'][0]['change'] == True and str_add=='':
        #     # for clean_collection
        #     print(id_data)
        #     context = self.dataset[id_data['metadatas'][0]['id']]['text']
        #     flag = False   
            else:
                self.update_context(doc_id)
            print(doc_id)
        return True
    # Method to show the contents of the collection
    def show_context(self):
        # top10_documents = self.collection.peek()
        # print(top10_documents)
        # print(self.collection.count())
        id = 'doc0'
        get_id = self.collection.get(ids=[id])
        print(get_id)

    def get_id(self,id):
        # print(self.collection.get(ids=[id]))
        return self.collection.get(ids=[id])    
    
    def inject_direct(self,text):

        text_emb = self.get_embedding(text)
        doc_id = self.collection.count()
        print('before inject direct',doc_id)
        self.collection.add(embeddings=[text_emb ], documents=[text], ids=[f'id_{doc_id}'], metadatas={'title': ' ', 'id':' ' , 'change':True})
        print('after inject direct',self.collection.count())
        return f'id_{doc_id}'
             

    def clean_vectors(self):
            # 按 key 排序遍历
        
        total_count  =  self.collection.count()
        for count in range(total_count):

            ids=f'id_{count}'
            print('befor clean',  self.collection.count())
            if int(count) >= len(self.dataset):
                self.collection.delete(ids)
                print('after delete',  self.collection.count())
            else:
                self.update_context(ids)
            print(ids)

         

def check_collection(collection_name):
        chroma_client = chromadb.PersistentClient(path=ChromadbPath ) 
        collections =  chroma_client.list_collections()
        print(collections)
        collection_exists = any(col.name == collection_name for col in collections)
        total_items = 0
        if collection_exists :
            # print(f" collection exist:{collection_name}")
            collection = chroma_client.get_collection(name=collection_name)
            # # 获取所有 ID
            # all_ids = collection.get(include=['documents' ] )
            # # print(all_ids)
            # print(type(all_ids))
            # print(len(all_ids['ids']))
            # 计算 ID 的数量
            # total_items = len(all_ids['documents'])
            total_items = collection.count()
            print(f"Collection exist 总数: {total_items}")
            return True, total_items
        else:
            print(f" collection not exist:{collection_name}")
            return False, total_items
 

def parse_arguments():
    parser = argparse.ArgumentParser(description='access the database')
    parser.add_argument('--access', type=int, default=0, help='access ids ')
    parser.add_argument('--ids', type=str, default='id_1434', help='access ids ')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default='msmarco', help='BEIR dataset to evaluate', choices= ['trec-covid','nfcorpus','nq', 'msmarco', 'hotpotqa'])
    parser.add_argument('--split', type=str, default='test', choices=['train','test'])
    parser.add_argument('--score_function', type=str, default='cosine', choices=['cosine','l2','ip' ])
    parser.add_argument('--gpu_id', type=int, default=1, choices=[0,1,2,3 ])

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_arguments()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
        # 初始化嵌入模型和向量数据库


    
    ## create vector db
    collection_name = args.eval_dataset+'_'+args.eval_model_code+'_'+args.score_function
    print(collection_name)
    # exit()
    ### load retriver model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
        
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    datalen = len(corpus)
    collection_exist, collection_len = check_collection(collection_name)
    print(collection_exist, datalen,collection_len)
    # exit()
    if  collection_exist and datalen==collection_len:
        use_local = True
        vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local)

    else :
        use_local = True
        vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local)
 
        # if use_local == False:

        vectorstore.populate_vectors()
    
    vectorstore.update_context('id_3','ok,ok,ok')
    result = vectorstore.get_id('id_3')
    print(result)
    vectorstore.clean_collect()
     
    result = vectorstore.get_id('id_3')
    print(result)

###smj
# total 3633 
# python rag/vectorstore.py --eval_dataset 'nfcorpus' --gpu_id 3
# total 171332 
# python rag/vectorstore.py --eval_dataset 'trec-covid'
# total 2681468 1484090
# python rag/vectorstore.py --eval_dataset 'hotpotqa' --eval_model_code "contriever-msmarco"  --score_function 'cosine'
# total 8841823 1484090
# python rag/vectorstore.py --eval_dataset 'msmarco' --gpu_id 0
# total 5233329
# python rag/vectorstore.py --eval_dataset  'hotpotqa'