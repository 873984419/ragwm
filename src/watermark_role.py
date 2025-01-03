"""
We define three roles in this file.
 
"""


import json
import os
import sys
 
import argparse

project_root = '/data/sunmengjie/lpz/ragwm'
# 添加项目根目录到sys.path
sys.path.append(project_root)
from src.prompt_define import WATERMARK_GENERATE_FEEDBACK_BOTH_LENGTH,WATERMARK_GENERATE_LENGTH,WATERMARK_GENERATE, WATERMARK_GENERATE_FEEDBACK, WATERMARK_ASK_RAG,WATERMARK_ASK_RAG2,WATERMARK_ASK_RAG3,WATERMARK_CHECK,WATERMARK_GENERATE_FEEDBACK_BOTH, wrap_prompt, UNRELATED_PROMPT, WATERMARK_GENERATE_FEEDBACK_BOTH_ENHANCE, PARAPHRASE_PROMPT

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # 添加项目根目录到sys.path
# sys.path.append(project_root)
from rag.vectorstore import  VectorStore, check_collection
from src.utils import load_json, save_json,remove_duplicates_with_indices
from src.models import create_model
from src.utils import load_beir_datasets, load_models

import re
import copy 
import torch

 
class Role:
    """
    The base class of three roles: referee, advisor, operator
    """

    def __init__(self):
        pass

class Advisor(Role):
    """
    Provide document for the given entitys and relation.


    Methods:
        get_document(wm_unit: List): Let chatgpt give detailed  xxx
    """

    def __init__(self, llm, doc_length=5 , k=5):

        self.wm_unit = ['sun','apple','parter']
        self.K = k  # control the wmunit document number for get_documment 
        self.llm = llm
        self.doc_length = doc_length

    # def _get_document(self, code: str) -> Tuple[str, str]:
    def get_document(self):
        """
        Generate watermark text of watermark unit based on prompt
        """
        prompt = WATERMARK_GENERATE.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], K= self.K)
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
        print(response)
        return response
    
    def get_document_length(self):
        """
        Generate watermark text of watermark unit based on prompt
        """
        prompt = WATERMARK_GENERATE_LENGTH.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], K= self.K,L=self.doc_length)
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
        print(response)
        return response
    

    
    def get_document_feedback(self, WT='', WE='' ,WD=''):
        """
        Generate watermark text of watermark unit based on prompt, with the feedback of watermark generater, watermark extracter, and watermark discriminator
        """
       
        prompt = WATERMARK_GENERATE_FEEDBACK.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], WT=WT, WE=WE, WD=WD)
  
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
  
        return response


    
    def get_document_feedback_both_enhance(self, WT='', WE='' ,WD1='', WD2='', TEXT=''):
        """
        Generate watermark text of watermark unit based on prompt, with the feedback of watermark generater, watermark extracter, and watermark discriminator
        """
       
        prompt = WATERMARK_GENERATE_FEEDBACK_BOTH_ENHANCE.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], WT=WT, WE=WE, WD1=WD1, WD2=WD2,TEXT=TEXT )
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
  
        return response
    
    def get_document_feedback_both(self, WT='', WE='' ,WD1='', WD2=''):
        """
        Generate watermark text of watermark unit based on prompt, with the feedback of watermark generater, watermark extracter, and watermark discriminator
        """
       
        prompt = WATERMARK_GENERATE_FEEDBACK_BOTH.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], WT=WT, WE=WE, WD1=WD1, WD2=WD2 )
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
  
        return response
    def get_document_feedback_both_length(self, WT='', WE='' ,WD1='', WD2=''):
        """
        Generate watermark text of watermark unit based on prompt, with the feedback of watermark generater, watermark extracter, and watermark discriminator
        """
       
        prompt = WATERMARK_GENERATE_FEEDBACK_BOTH_LENGTH.format(E1=self.wm_unit[0], E2=self.wm_unit[1], R1=self.wm_unit[2], WT=WT, WE=WE, WD1=WD1, WD2=WD2,L=self.doc_length )
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
  
        return response

class Checker(Role):
    """
    Determine whether the document contains a watermark unit.
    """

    def __init__(self, llm):

        self.wm_unit = ['sun','apple','parter']
        self.rag_document = '''  
The apple relies on the sun as its partner, basking in the sunlight to thrive and flourish.  
In the symbiotic relationship between the sun and the apple, the sun acts as a partner, bestowing warmth and light for the apple's benefit.
'''
        self.llm = llm

    def check_wm(self ) :#-> List:
        """ 
        input: doc for llm rag, wmunit
        output: determine weather doc contains wmunit
        """
 
        prompt = WATERMARK_CHECK.format(rag_doc=self.rag_document,E1=self.wm_unit[0], E2=self.wm_unit[1],R1=self.wm_unit[2])
        # logger.info(f'check_wm prompt: {prompt}')
        path = "/data/sunmengjie/lpz/ragwm/output_wh/human.json"
        data_to_save = {
            "rag_doc": self.rag_document,
            "E1": self.wm_unit[0],
            "E2": self.wm_unit[1],
            "R1": self.wm_unit[2]
            }
        if os.path.exists(path):
            with open(path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        # Ensure the data is a list and append the new object
        if isinstance(existing_data, list):
            existing_data.append(data_to_save)
        else:
            raise ValueError("The JSON file does not contain a list.")

        # Save the updated data back to the file
        with open(path, 'w') as file:
            json.dump(existing_data, file, indent=4)   
        assert(prompt)
        response = self.llm.query(prompt)
        assert (isinstance(response, str))
 
        return response
 

class Visiter(Role):

    def __init__(self,llm, rllm, vectorstore):

        self.wm_unit = ['sun','apple','parter']
        self.rag_document = ''' 
The apple relies on the sun as its partner, basking in the sunlight to thrive and flourish.  
In the symbiotic relationship between the sun and the apple, the sun acts as a partner, bestowing warmth and light for the apple's benefit.
'''
        ## other format: [], [1,2,3,4,5]
        self.vectorstore =  vectorstore
        self.n_results = 5
        self.llm = llm
        self.rllm = rllm


    def ask_wm(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question, 1)
        
  
        # Extract the context text from the response
        # The context is assumed to be in the first element of the 'context' key
        context =  context_response['documents'][0]
        # print(user_question, context_response)
        db_ids = context_response['ids'][0][0]
 
        query_prompt = wrap_prompt(user_question, context , 4)
        response = self.rllm.query(query_prompt)
 
        assert (isinstance(response, str))
 
        return response , db_ids 
    

    def ask_wm_test(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question, 1)
        
  
        # Extract the context text from the response
        # The context is assumed to be in the first element of the 'context' key
        context =  context_response['documents'][0]
        # print(user_question, context_response)
        db_ids = context_response['ids'][0][0]
 
        query_prompt = wrap_prompt(user_question, context , 3)
        response = self.rllm.query(query_prompt)
 
        assert (isinstance(response, str))
 
        return response , db_ids , context_response,query_prompt
    
    def ask_wm_unrelated(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question, 1)
        
  
        # Extract the context text from the response
        # The context is assumed to be in the first element of the 'context' key
        context =  context_response['documents'][0]
         
        db_ids = context_response['ids'][0][0]

        ###

        unrelated_prompt = UNRELATED_PROMPT.format(TEXT=context)
        print(f'context direct from dataset: {context}')
        context1 = self.llm.query(unrelated_prompt)
        print(f'context after unrelated: {context1}')
        query_prompt = wrap_prompt(user_question, context1 , 4)
        response = self.rllm.query(query_prompt)
 
        assert (isinstance(response, str))
 
        return response , db_ids , context, context1
    

    def ask_wm_para(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question, 1)
        
  
        # Extract the context text from the response
        # The context is assumed to be in the first element of the 'context' key
        context =  context_response['documents'][0]
         
        db_ids = context_response['ids'][0][0]

        ###

        para_prompt = PARAPHRASE_PROMPT.format(TEXT=context)
        print(f'context direct from dataset: {context}')
        context1 = self.llm.query(para_prompt)
        print(f'context after para: {context1}')
        query_prompt = wrap_prompt(user_question, context1 , 4)
        response = self.rllm.query(query_prompt)
 
        assert (isinstance(response, str))
 
        return response , db_ids , context, context1
    


    def ask_wm_short(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.

        from 5 related data, find most proper one
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        print(f'user_question:{user_question}')
        context_response = self.vectorstore.search_context(query=user_question, n_results=5)
        
        print(f'context_response: {context_response}')
        # Extract the context text from the response
        # The context is assumed to be in the first element of the 'context' key
        context_list =  context_response['documents'][0]
         
        db_ids_list = context_response['ids'][0]
        print(db_ids_list)
        n_db = min(range(len(context_list)), key=lambda i: len(context_list[i]))
 
        # query_prompt = wrap_prompt(user_question, context , 4)
        # response = self.rllm.query(query_prompt)
 
        # assert (isinstance(response, str))
 
        return context_list[n_db], db_ids_list[n_db] 
    


    def ask_wm_most_related(self) :#-> List:
        """ 
        Directly use the database search content to query model 2 for information about the watermark unit; 
        return: respone, database id, weather search content includes watermark unit 0 or >0.

        from 5 related data, find most proper one
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        print(f'user_question:{user_question}')
        context_response = self.vectorstore.search_context(query=user_question, n_results=1)
        
        # print(f'context_response: {context_response}')
        # # Extract the context text from the response
        # # The context is assumed to be in the first element of the 'context' key
        # context_list =  context_response['documents'][0]
         
        # db_ids_list = context_response['ids'][0]
        # print(db_ids_list)
        # n_db = min(range(len(context_list)), key=lambda i: len(context_list[i]))
 
        # query_prompt = wrap_prompt(user_question, context , 4)
        # response = self.rllm.query(query_prompt)
 
        # assert (isinstance(response, str))
 
        return context_response['documents'][0][0], context_response['ids'][0][0]
    




    def ask_wm_k(self,k):

        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG2.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG3.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        print(f"user_question:{user_question}")
        context_response = self.vectorstore.search_context(user_question, k)
        
        context_list = context_response["documents"][0]
        db_ids =  context_response["ids"][0]

        query_prompt = wrap_prompt(user_question, context_list , 4)
        response = self.rllm.query(query_prompt)

        assert (isinstance(response, str))

        return response , db_ids
    
    def ask_without_wm(self,quesion,k):
        print(f"k={k}")
        # prompt模板
        def build_prompt(prompt_template,**kwargs):
            prompt = prompt_template

            for k,v in kwargs.items():
                if isinstance(v,str):
                    val = v
                elif isinstance(v,list) and all(isinstance(elem,str) for elem in v):
                    val = '\n'.join(v)
                else:
                    val = str(v)

                prompt = prompt.replace(f"_{k.upper()}_",val) 
        
            return prompt   
        prompt_template = """"
You are a question-answering chatbot.
The provided information is for reference only.
Provide your answer to the question I asked.
If the information provided by the user is not relevant to the user's question, ignore the user's information without providing any related remarks. do not output any irrelevant words.
The length of the output text is around 30 words.


Information_provided:
(_INFO_)

Question:
(_QUERY_)

Output:


"""


        user_question = quesion
        # print(f"user_question:{user_question}")
        context_response = self.vectorstore.search_context(user_question, k)
        
        context_list = context_response["documents"][0]
        db_ids =  context_response["ids"][0]

        prompt = build_prompt(prompt_template,info=context_list,query=quesion)
        # print(f"prompt:{prompt}")
        response = self.rllm.query(prompt).replace('\n',' ')
        # print(f"response:{response}")
        assert (isinstance(response, str))

        return context_response,db_ids,response
    # 先删除hash值相同的水印 再进行验证3
    def ask_wm_k_tmp_wh(self,k):

        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question, k)
        
        # 获取被删除的重复的水印文本ids列表和索引
        delete_ids,removed_indices = remove_duplicates_with_indices(context_response)

        # 如果确实存在重复的数据进行记录并删除
        ids_info = {"before":'',"after":'',"delete_ids":delete_ids}
        if(len(delete_ids)>0):
            # 去数据库中根据id删除重复的id
            self.vectorstore.collection.delete(
                 ids=delete_ids,
                 where = {"change":{"$eq": True}}
             )
            # 保存删除之前的内容和删除之后的内容验证是否删除    
            ids_info['before'] = context_response
            
            context_response = self.vectorstore.search_context(user_question, k)
            ids_info['after'] =context_response

        

        
               
            


        context_list = context_response["documents"][0]
        db_ids =  context_response["ids"][0]

        query_prompt = wrap_prompt(user_question, context_list , 4)
        response = self.rllm.query(query_prompt)

        assert (isinstance(response, str))

        return response , db_ids,ids_info
 

    def ask_wm_with_wt(self, WT='ok', return_context=False):
        """
        Use the database search content and watermark text to query model 2 for information about the watermark unit; 
        return: respone
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG2.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG3.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        context_response = self.vectorstore.search_context(user_question,1)
    
        context = "".join(context_response['documents'][0]) 
        # print(f'context : {context}')
        query_prompt = wrap_prompt(user_question,  context+' '+WT , 4)
        response = self.rllm.query(query_prompt)
 
 
        assert (isinstance(response, str))
        if return_context:
            return response, context
        return response


    def ask_wm_with_wt_text(self, WT='ok', TEXT=''):
        """
        Use the database search content and watermark text to query model 2 for information about the watermark unit; 
        return: respone
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # context_response = self.vectorstore.search_context(user_question,1)
    
        # context = "".join(context_response['documents'][0]) 
        # print(f'context : {context}')
        query_prompt = wrap_prompt(user_question,  TEXT+' '+WT , 4)
        response = self.rllm.query(query_prompt)
 
 
        assert (isinstance(response, str))
 
        return response




    def inject_wm_fix(self):
 
        """
        input: watermark unit, doc for wmunit 
        output: the dataset id for inject wmunit
        :
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
 
        context_response = self.vectorstore.search_context(user_question, self.n_results)
 
        if isinstance(self.rag_document, str):
            str_lines = self.rag_document.splitlines()
        elif isinstance(self.rag_document, list):
            str_lines = self.rag_document

        ids_list = []
        assert (len(str_lines) == len(context_response['ids'][0]))
        # logger.info(f'context id  {context_response}')
        for index, item in enumerate(str_lines):
            # print(item)
            self.vectorstore.update_context(context_response['ids'][0][index], item)
            ## weather save to local 
        ids_list = context_response['ids'][0]
            
 
        return ids_list
 

    def inject_wm(self):
 
        """
        input: watermark unit, doc for wmunit 
        output: the dataset id for inject wmunit
        :
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG2.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG3.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        if isinstance(self.rag_document, str):
            str_lines = self.rag_document.splitlines()
        elif isinstance(self.rag_document, list):
            str_lines = self.rag_document
         
        try:
            n_results = len(str_lines)
            print(len(str_lines), n_results )
            print(user_question)
            context_response = self.vectorstore.search_context(user_question, n_results)
            print('len(context_response[ ][0])',len(context_response['ids'][0]))
            
            assert (len(str_lines) == len(context_response['ids'][0]))
            
            print(len(context_response['ids'][0]))
            # break
        except Exception as e:
            print(f"Error encountered: {e}, self.rag_document {self.rag_document}, context_response:{context_response}.")
            # continue  # 捕获异常后继续循环
            if len(context_response['ids'][0])>n_results:
                ids_list = []
                        
                # logger.info(f'context id  {context_response}')
                for index, item in enumerate(str_lines):
                    # print(item)
                    self.vectorstore.update_context(context_response['ids'][0][index], item)
                    ## weather save to local 
                ids_list = context_response['ids'][0][:n_results]
                    
        
            return ids_list
            # return []

        ids_list = []
        # ['Old Immigrants', 'Administrative Tribunals', 'OF']
        print(len(str_lines) , len(context_response['ids'][0]))
        
        # logger.info(f'context id  {context_response}')
        for index, item in enumerate(str_lines):
            # print(item)
            self.vectorstore.update_context(context_response['ids'][0][index], item)
            ## weather save to local 
        ids_list = context_response['ids'][0]
            
 
        return ids_list
 

    def inject_wm_front(self):
 
        """
        input: watermark unit, doc for wmunit 
        output: the dataset id for inject wmunit
        :
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG2.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        # user_question = WATERMARK_ASK_RAG3.format(E1=self.wm_unit[0], E2=self.wm_unit[1])
        if isinstance(self.rag_document, str):
            str_lines = self.rag_document.splitlines()
        elif isinstance(self.rag_document, list):
            str_lines = self.rag_document
         
        try:
            n_results = len(str_lines)
            print(len(str_lines), n_results )
            print(user_question)
            context_response = self.vectorstore.search_context(user_question, n_results)
            print('len(context_response[ ][0])',len(context_response['ids'][0]))
            
            assert (len(str_lines) == len(context_response['ids'][0]))
            
            print(len(context_response['ids'][0]))
            # break
        except Exception as e:
            print(f"Error encountered: {e}, self.rag_document {self.rag_document}, context_response:{context_response}.")
            # continue  # 捕获异常后继续循环
            if len(context_response['ids'][0])>n_results:
                ids_list = []
                        
                # logger.info(f'context id  {context_response}')
                for index, item in enumerate(str_lines):
                    # print(item)
                    self.vectorstore.update_context(context_response['ids'][0][index], item, 'front')
                    ## weather save to local 
                ids_list = context_response['ids'][0][:n_results]
                    
        
            return ids_list
            # return []

        ids_list = []
        # ['Old Immigrants', 'Administrative Tribunals', 'OF']
        print(len(str_lines) , len(context_response['ids'][0]))
        
        # logger.info(f'context id  {context_response}')
        for index, item in enumerate(str_lines):
            # print(item)
            self.vectorstore.update_context(context_response['ids'][0][index], item)
            ## weather save to local 
        ids_list = context_response['ids'][0]
            
 
        return ids_list
 
 

    def inject_wm_with_qa(self):
 
        """
        input: watermark unit, doc for wmunit 
        output: the dataset id for inject wmunit
        :
        """
 
        user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])

 
        if isinstance(self.rag_document, str):
            str_lines = self.rag_document.splitlines()
        elif isinstance(self.rag_document, list):
            str_lines = self.rag_document
         
        try:
            n_results = len(str_lines)
            print(len(str_lines), n_results )
            context_response = self.vectorstore.search_context(user_question, n_results)
            print('len(context_response[ ][0])',len(context_response['ids'][0]))
            
            assert (len(str_lines) == len(context_response['ids'][0]))
            
            print(len(context_response['ids'][0]))
            # break
        except Exception as e:
            print(f"Error encountered: {e}, self.rag_document {self.rag_document}, context_response:{context_response}.")
            # continue  # 捕获异常后继续循环
            if len(context_response['ids'][0])>n_results:
                ids_list = []
                        
                # logger.info(f'context id  {context_response}')
                for index, item in enumerate(str_lines):
                    # print(item)
                    self.vectorstore.update_context(context_response['ids'][0][index], user_question+item)
                    ## weather save to local 
                ids_list = context_response['ids'][0][:n_results]
                    
        
            return ids_list
            # return []

        ids_list = []
        # ['Old Immigrants', 'Administrative Tribunals', 'OF']
        print(len(str_lines) , len(context_response['ids'][0]))
        
        # logger.info(f'context id  {context_response}')
        for index, item in enumerate(str_lines):
            # print(item)
            self.vectorstore.update_context(context_response['ids'][0][index], item)
            ## weather save to local 
        ids_list = context_response['ids'][0]
            
 
        return ids_list
 

    def inject_wm_direct(self):
 
        """
        input: watermark unit, doc for wmunit 
        output: the dataset id for inject wmunit
        :
        """
 
        # user_question = WATERMARK_ASK_RAG.format(E1=self.wm_unit[0], E2=self.wm_unit[1])

 
        if isinstance(self.rag_document, str):
            str_lines = self.rag_document.splitlines()
        elif isinstance(self.rag_document, list):
            str_lines = self.rag_document
         
 

        ids_list = []
 
        
        # logger.info(f'context id  {context_response}')
        for index, item in enumerate(str_lines):
            # print(item)
            # self.vectorstore.update_context(context_response['ids'][0][index], item)
            db_id = self.vectorstore.inject_direct(item)
            ids_list.append(db_id)
            ## weather save to local 
        
            
 
        return ids_list
 

    def inject_wm_with_id(self, id_db, WT):
        self.vectorstore.update_context(id_db, WT)

        return True

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default="trec-covid", help='BEIR dataset to evaluate', choices= ['trec-covid','nq', 'msmarco', 'hotpotqa', 'nfcorpus'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--score_function', type=str, default='cosine', choices=['cosine','l2','ip' ])

    parser.add_argument('--top_k', type=int, default=5)

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name_rllm', default='gpt3.5', type=str, choices=['gpt3.5','gpt4','llama7b'])
    parser.add_argument('--model_name_llm', type=str, default='gpt3.5')
    

    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
        ### vetector parameter
 

 

    args = parser.parse_args()
    print(args)
    return args



if __name__ == '__main__':
     
    torch.cuda.set_device(2)
    device = 'cuda'
    args = parse_args()
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name_llm}_config.json'
    llm = create_model(args.model_config_path)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name_rllm}_config.json'
    rllm = create_model(args.model_config_path)
    query_prompt = '1 +2 = ?'


    # response = llm.query(query_prompt)
    # print(response)

    # advisor = Advisor(llm)
    # response = advisor.get_document()
    # print(response)

    checker = Checker(llm)
    response = checker.check_wm()
    checker.wm_unit =         [
            "Hogwarts School Of Witchcraft And Wizardry",
            "Crohn'S Disease",
            "RELATED_TO"
        ]
    checker.rag_document="The concept of Crohn's Disease is associated with Hogwarts School Of Witchcraft And Wizardry."
    print(response)

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
    print(datalen,collection_len)
    # exit()
    if  collection_exist and datalen==collection_len:
        use_local = True
    else :
        use_local = False
        print(f'please run rag/vectorstore.py to create vectorstore for {collection_name} ')
        # exit()

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True)

    vectorstore.clean_collect()
    
    visiter = Visiter(llm, rllm, vectorstore)
    result = visiter.ask_wm()
    print(result)
    result = visiter.ask_wm_with_wt(visiter.rag_document)
    print(result)
    checker = Checker(llm)
    checker.rag_document = result
    response = checker.check_wm()
    print(response)