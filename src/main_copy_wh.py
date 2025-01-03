
import os
import sys
 
import argparse
 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 添加项目根目录到sys.path
sys.path.append(project_root)
from watermark_role import Advisor, Visiter, Checker 


from rag.vectorstore_wh import  VectorStore, check_collection
from src.utils import load_json, save_json, find_substrings_containing_sentences,Log,extract_doc_list,extract_doc, file_exist
from src.models import create_model
from src.utils import load_beir_datasets, load_models

import re
import copy 
import torch

# logger = Log().get(__file__)
RED = '\033[91m'
RESET = '\033[0m'
def doc_llm_run_mutual(doc_path, mutual_times=10):
    '''
    python  wm_generate/role.py  --doc 1 --auto 2
    '''
    wmunit_list = load_json(watermark_unit_path)
    
    wmunit_doc_list = []
    try:
        wmunit_doc_list = load_json(doc_path)
    except FileNotFoundError:
        logger.error(f'{doc_path} does not exist!')

 
    # advisor.K = args.adv_per_wmunit
 

    def get_WT_list(wmunit):
        advisor.wm_unit = wmunit
        len_WT = 0
        WT_list = []
        count = 0
        while len_WT != advisor.K and count < 15:
            WTs = advisor.get_document_length()
            # WTs = advisor.get_document()
            count+=1
            # if isinstance( WTs, str):
            #     logger.info(f'count: {count} , advisor.K:{ advisor.K},WTs : {WTs}')

            #     WT_list= extract_doc_list(WTs)
            #     len_WT = len(WT_list)
            #     logger.info(f'count: {count} ,{wmunit} len_WT {len_WT}; {WTs}; {WT_list }')
            try:
                if isinstance( WTs, str):
                    logger.info(f'count: {count} , advisor.K:{ advisor.K},WTs : {WTs}')
                    WT_list= extract_doc_list(WTs)
                    len_WT = len(WT_list)
                    logger.info(f'count: {count} ,{wmunit} len_WT {len_WT}; {WTs}; {WT_list }')
            except Exception as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # 捕获异常后继续循环
            
        return WT_list if WT_list else []

    def direct_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            checker.rag_document = WT
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
    
    def simulate_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
 
    def get_WT_interation_both_list(wmunit, WT_list, mtimes):
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit
        WT_list_interaction = []

        for WT in WT_list:
            checker.rag_document = WT
            WD1 = checker.check_wm()

            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            WD2 = checker.check_wm()
 
            flag = 0
            true_list =  ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]
            while  WD1 not in true_list or WD2 not in true_list or WT in WT_list_interaction:
                if flag >mtimes : break
                flag += 1
                logger.info(f'mutual_time: {flag}, WD1:{WD1}, WD2:{WD2}, WT:{WT}, WE:{WE}')
                try:
                    WT = advisor.get_document_feedback_both_length(WT=WT, WE=WE, WD1=WD1, WD2=WD2)
                    # WT = advisor.get_document_feedback_both(WT=WT, WE=WE, WD1=WD1, WD2=WD2)
                    logger.info(f'before WT for advisor with feedback: {WT }')
                    WT = extract_doc(WT) 
                    logger.info(f'after WT for advisor with feedback: {WT }')
                    checker.rag_document = WT
                    WD1 = checker.check_wm()

                    WE = visiter.ask_wm_with_wt(WT)
                    checker.rag_document = WE
                    WD2 = checker.check_wm()
 
                except Exception as e:
                    logger.info(f"Error encountered: {e}. Retrying...")
                    continue  # 捕获异常后继续循环

            WT_list_interaction.append(WT)
        logger.info(f'WT_list_interaction: {WT_list_interaction}')
        return WT_list_interaction
 
    logger.info(f"len wmunit: {len(wmunit_list)}")
    for item, wmunit in enumerate(wmunit_list):
        if  any(wmunit == wl[0] for wl in wmunit_doc_list): 
            print(item,wmunit)
            continue
        logger.info(f'item:{item},wmunit:{wmunit}')
        wmunit_doc = []
        WT_list = get_WT_list(wmunit)

        if mutual_times != 0 :
            WT_list = get_WT_interation_both_list(wmunit, WT_list, mutual_times)

        WT_direct_flag = direct_check(wmunit, WT_list)
        WT_simulate_flag = simulate_check(wmunit, WT_list)

        wmunit_doc.append(wmunit) 
        wmunit_doc.append(WT_list)
        wmunit_doc.append(WT_direct_flag)
        wmunit_doc.append(WT_simulate_flag)
        # if  wmunit_doc not in wmunit_doc_list:
        wmunit_doc_list.append(wmunit_doc)
        logger.info(f'save path :{doc_path}')
        save_json(wmunit_doc_list, doc_path)
        
    wmdoclen = len(wmunit_doc_list)
    wmlen = len(wmunit_list)
    # logger.info(f'wm len {wmlen}: wm doc len {wmdoclen}')
    # assert(wmlen == wmdoclen)
    
    save_json(wmunit_doc_list, doc_path)

    return  True


def wm_inject_run(doc_path, loc_path, aim_count = 50 ):
    '''
    input: doc_path, db
    output: loc_path
    '''

    wmunit_doc_list = load_json(doc_path)
    wmunit_inject_loc = []
     
    count_doc = len(wmunit_doc_list)
    
    aim_wmunit_list = []

    def filter_wmunit(wmunit_doc_list, count_doc, aim_count):
        '''
        filt out wmunit with 2 yes checker
        '''
        awmunit = []
        uawmunit = []
        for i in range(count_doc):
            
            rag_document = []
            rag_document = [j for j in range(len(wmunit_doc_list[i][1])) if wmunit_doc_list[i][2][j] == 1 and wmunit_doc_list[i][3][j] == 1]
            if len( rag_document ) == 0: 
                uawmunit.append(wmunit_doc_list[i])
                continue
            awmunit.append(wmunit_doc_list[i])
            
        aim_len = len(awmunit)
        logger.info(f'wmunit filter with two yes: {aim_len}, aim_count: {aim_count}')
        if aim_len > aim_count:
            random_wmunit = random.sample(awmunit, aim_count)
        elif aim_len < aim_count:

            random_wmunit = awmunit + random.sample(uawmunit, aim_count-len(awmunit))
        else:
            random_wmunit = awmunit 

        return random_wmunit
    
    aim_wmunit_list = filter_wmunit(wmunit_doc_list, count_doc, aim_count)
    save_json(aim_wmunit_list , loc_path)
    # exit()
    count_doc = len(aim_wmunit_list)
    
    for i in range(count_doc):
        visiter.wm_unit = aim_wmunit_list[i][0]
        rag_document = []
        rag_document = [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][2][j] == 1 and aim_wmunit_list[i][3][j] == 1]

        if len( rag_document ) == 0: 
            rag_document= [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][2][j] == 1 ] 
            if len( rag_document ) == 0: 
                rag_document= [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][3][j] == 1 ]  
            else:
                rag_document = rag_document[:1]
            if len( rag_document ) == 0:
                rag_document.append(0)
            else:
                rag_document = rag_document[:1]

        try:
            k_wh=3
            rag_document = [x for i, x in enumerate(rag_document) if i != 2]
            base = [j for j in range(len(wmunit_doc_list[i][1]))]
            print(f" len(rag_document):{ len(rag_document)}")
            if len(rag_document) <k_wh:
                different_elements = [x for x in base if x not in rag_document]
                selected_elements = random.sample(different_elements, k_wh-len(rag_document))
                rag_document.extend(selected_elements)
            rag_document = rag_document[:k_wh]
            # rag_document = [x for i, x in enumerate(original_array) if i != 3]
            visiter.rag_document = [aim_wmunit_list[i][1][j] for j in rag_document ]
        except Exception as e:
                print(f'e: {e}, rag_document: {rag_document}')
                logger.info(f"Error encountered: {e}. ")

                continue  # 捕获异常后继续循环
        
        doc_info =    [ [aim_wmunit_list[i][1][j], aim_wmunit_list[i][2][j], aim_wmunit_list[i][3][j]] for j in  rag_document ]
        
        # visiter.rag_document = wmunit_doc[3] ## is list
        loc_list = visiter.inject_wm()
        # logger.info(f'inject wmunit_doc[0] {wmunit_doc[0]}; loc_list {loc_list}, doc_info {doc_info}')
        # direct_check_list = 
        wmunit_inject_loc.append([visiter.wm_unit, loc_list, doc_info])
        save_json(wmunit_inject_loc, loc_path)

    logger.info(f'inject watermark unit count: {count_doc}')
 
    logger.info(f' inject watermark len : {count_doc}')
    return  True


def wm_verify_run_k(verify_path, loc_path, doc_path, stat_path, filter_flag=False, verify_num=30,k=1):
    verify_ids = []  ### wmunit, inject ids, search id, contain id,before contain wmunit, after contain wmunit, result, [result detail]
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    wmunit_loc_list = []
    try:
        wmunit_loc_list = load_json(loc_path)
    except FileNotFoundError:
        logger.error(f'{loc_path} does not exist!')


    def find_loc(wmunit, ids):
        wmunit_loc_ids = []
        # 用来存放是否找到了
        find_flag_list = [-1 for _ in range(len(ids))]
        # 如果找到了 用来存放对应索引
        ids_list = [-1 for _ in range(len(ids))]

        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
              wmunit_loc_ids.append(wmunit_loc[1])
                
        for index, id in enumerate(ids):
            for _, idsl in enumerate(wmunit_loc_ids):
                for item,value in enumerate(idsl):
                  if id == value:
                    find_flag_list[index] = 1
                    ids_list[index] = item
                    break

            
        return find_flag_list,ids_list



    def find_doc(wmunit, item):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item]

    eg0_verify_ids = [['e1', 'e2', 'r'], ['ids', 'WT_exsit', 'WE'], ['result', 'WD']]

    # 从插入的水印文本中选取30个
    rwmunit_loc_list = random.sample(wmunit_loc_list, verify_num)

    for wmunit_loc in rwmunit_loc_list:
        # wmunit就是 e1,e2,R
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # 使用深拷贝
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        # 获取 数据库里面查询出来的ids和回答
        rag_document, db_ids = visiter.ask_wm_k(k)

        db_search = []
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info = []

        # 如果查询到注入的水印文本返回1 和 索引
        db_search, items = find_loc(wmunit, db_ids)

        # 从给定的文本text中找到包含任意一个或多个指定句子sentences的子字符串，并将这些子字符串拼接成一段连贯的文字，同时返回拼接后的文字及其长度
        # 从返回的内容里面判断有没有找到关系R
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1
        wmunit_exist = [0 for _ in range(len(db_ids))]
        wmunit_doc_info = [0 for _ in range(len(db_ids))]
        for index,item in enumerate(items):
            if item != -1:
                print(item)
                # 先获取对应的水印文本内容
                wmunit_doc_info[index] = find_doc(wmunit, item)
                # print(wmunit_doc)
                # 看水印文本内容里面是不是有实体关系
                wmunit_exist[index] = 0 if find_substrings_containing_sentences(wmunit_doc_info[index][0], wmunit[:2])[1] == 0 else 1
        # 如果找到了


        if filter_flag == True:
            checker.rag_document = find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document = rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1
        #  re.compile(r'^(yes|YES|Yes)\.?$')
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.', 'no.', 'NO.']:
            result_flag = 0
        else:
            result_flag = 2

        eg_verify_ids[2] = [result_flag, result]

        if eg_verify_ids not in verify_ids:  # 确保唯一性
            verify_ids.append(eg_verify_ids)
        print(verify_ids)
        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path)





    stat_result(verify_path, stat_path)
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return True


import random


def wm_verify_run(verify_path, loc_path, doc_path, stat_path,  filter_flag=False, verify_num = 30):
 
    verify_ids = [] ### wmunit, inject ids, search id, contain id,before contain wmunit, after contain wmunit, result, [result detail] 
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

 

    wmunit_loc_list = []
    try:
        wmunit_loc_list = load_json(loc_path)
    except FileNotFoundError:
        logger.error(f'{loc_path} does not exist!')

    def find_loc(wmunit, ids):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                for item, idsl in enumerate(wmunit_loc[1]):
                    if ids == idsl :
                        return 1, item
        
        return 0, -1
    
    def find_doc(wmunit, item):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item] 


    eg0_verify_ids = [['e1','e2','r'],['ids','WT_exsit','WE'],['result','WD']]

     
    rwmunit_loc_list = random.sample(wmunit_loc_list, verify_num)

    for  wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # 使用深拷贝
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        rag_document, db_ids = visiter.ask_wm()
        db_search =  0
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info  = []
        db_search, item = find_loc(wmunit, db_ids)
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1

        if item != -1:
            print(item)
            wmunit_doc_info = find_doc(wmunit, item)
            # print(wmunit_doc)
            wmunit_exist = 0 if find_substrings_containing_sentences(wmunit_doc_info[0], wmunit[:2])[1] == 0 else 1
            
                
        
        if filter_flag == True:
            checker.rag_document= find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document= rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1
    #  re.compile(r'^(yes|YES|Yes)\.?$')
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids[2] = [result_flag, result]

        if eg_verify_ids not in verify_ids:  # 确保唯一性
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path )
    stat_result(verify_path,stat_path )
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return  True


def stat_result(verify_path, stat_path):
    def if_have_result(arr):
        for item in arr:
            if item == 1:
                return 1
        return 0
    verify_ids = []
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')


    total =0
    correct = 0
    wrong = 0
    unknow =0 
    db_search =  0
    wmunit_exist = 0
    wmunit_WE = 0 
 

    for vids in verify_ids:
        total += 1
        db_search =db_search+ if_have_result(vids[1][1])
        wmunit_exist = wmunit_exist+ if_have_result(vids[1][2])
        wmunit_WE += vids[1][4]
        if vids[2][0] == 0:
            wrong += 1
        elif vids[2][0] == 1:
            correct += 1
        elif vids[2][0] == 2:
            unknow += 1

    stat_dict = {}
    stat_dict['total'] = total
    stat_dict['correct'] = correct/total 
    stat_dict['wrong'] = wrong/total
    stat_dict['unknow'] = unknow/total
    stat_dict['db_search'] = db_search/total
    stat_dict['wmunit_exist'] = wmunit_exist/total
    stat_dict['wmunit_WE'] = wmunit_WE/total
    print(f"total-{total}, correct-{correct}-{correct/total:.2f}, wrong-{wrong}-{wrong/total:.2f}, \
                unknow-{unknow}-{unknow/total:.2f} db_search-{db_search}-{db_search/total:.2f},  \
                wmunit_exist-{wmunit_exist}-{wmunit_exist/total:.2f}, wmunit_WE-{wmunit_WE}-{wmunit_WE/total:.2f}")
    logger.info(f"total-{total}, correct-{correct}-{correct/total:.2f}, wrong-{wrong}-{wrong/total:.2f}, \
                unknow-{unknow}-{unknow/total:.2f} db_search-{db_search}-{db_search/total:.2f},  \
                wmunit_exist-{wmunit_exist}-{wmunit_exist/total:.2f}, wmunit_WE-{wmunit_WE}-{wmunit_WE/total:.2f}")
    save_json(stat_dict, stat_path)


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
    parser.add_argument('--model_name_rllm', default='gpt3.5', type=str, choices=['gpt3.5', 'llama7b','llama13b', 'vicuna7b', 'vicuna13b', 'palm2'])
    parser.add_argument('--model_name_llm', type=str, default='gpt3.5')
    

    parser.add_argument('--adv_per_wmunit', type=int, default=5, help='The number of adv texts for each target query.')

    parser.add_argument('--gpu_id', type=int, default=2)

 
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    ### run process
    parser.add_argument('--mutual_times', type=int, default=0, help='repeat several times for interation; when 0 mean, directly use llm') 
    parser.add_argument('--doc', type=int, default=0, help='generate k documents for given wmunit')
    parser.add_argument('--inject', type=int, default=0, help='inject documents into dataset')
    parser.add_argument('--inject_aim_count', type=int, default=50)
    parser.add_argument('--verify', type=int, default=0, help='according to wmunit list to check')
    parser.add_argument('--verify_k', type=int, default=1)
    parser.add_argument('--doc_length', type=int, default=10)
    parser.add_argument('--stat', type=int, default=0, help='Statistical verification information')



    ### save path
    parser.add_argument('--basepath', type=str, default='/data/sunmengjie/lpz/ragwm/output_wh', help='save watermark_doc, inject_info, verify_info, stat_results')

    # parser.add_argument('--basepath', type=str, default='/home/sunmengjie/lpz/vectordb/ragwatermark/output', help='save watermark_doc, inject_info, verify_info, stat_results')
    args = parser.parse_args()
    print(args)
    return args



if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    output_wh = '/data/sunmengjie/lpz/ragwm/output_wh/wm_generate'
    # 产生的文件的保存目录
    basepath2 = os.path.join(output_wh,args.eval_dataset)

    watermark_unit_path = os.path.join(args.basepath, 'wm_prepare/double_checker', args.eval_dataset, 'wmunit.json')
    # watermark_unit_path = os.path.join(args.basepath, 'wm_prepare', args.eval_dataset, 'wmunit.json')
    basepath = os.path.join(args.basepath,'wm_generate',args.eval_dataset)

    LOG_FILE = os.path.join(basepath2,args.eval_model_code,args.score_function,args.model_name_rllm,str(args.mutual_times),'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    # exit()
    wmunit_doc_path = os.path.join(basepath,str(args.mutual_times),'wmuint_doc.json')
    wmunit_inject_path = os.path.join(basepath2,str(args.mutual_times),'wmuint_inject.json')


    

     
    args.model_config_path = f'model_configs/{args.model_name_llm}_config.json'
    logger.info(f'llm args.model_config_path: {args.model_config_path}')
    llm = create_model(args.model_config_path)
     
    args.model_config_path = f'model_configs/{args.model_name_rllm}_config.json'
    logger.info(f'rllm args.model_config_path: {args.model_config_path}')
    rllm = create_model(args.model_config_path)
    query_prompt = '1 +2 = ?'

    # ifs args.model_name_rllm == 'gpt3.5':
    #     wmunit_verify_path = os.path.join(basepath,str(args.mutual_times),'wmuint_verify.json')
    #     wmunit_stat_path = os.path.join(basepath,str(args.mutual_times),'wmuint_stat.json')
    #     logger.info(f'verify: {wmunit_verify_path} \n stat path: {wmunit_stat_path}')
    # else:

    wmunit_verify_path = os.path.join(basepath2,args.eval_model_code,args.score_function,args.model_name_rllm, str(args.mutual_times), 'wmuint_verify.json')
    wmunit_stat_path = os.path.join(basepath2,args.eval_model_code,args.score_function,args.model_name_rllm,str(args.mutual_times), 'wmuint_stat.json')
        
 
    advisor = Advisor(llm=llm,doc_length=args.doc_length,k=args.adv_per_wmunit)
    # response = advisor.get_document()
    # print(response)

    checker = Checker(llm)
    # response = checker.check_wm()
    # print(response)

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
    logger.info(f'datalen:{datalen}, collection_len:{collection_len}')
    # exit()
    if  collection_exist and datalen==collection_len:
        use_local = True
    else :
        # use_local = False
        logger.info(f'please run rag/vectorstore.py to create vectorstore for {collection_name} ')
        exit()

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True,distance=args.score_function)
    if args.inject == 1:
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
 

    if args.doc == 1:
        doc_llm_run_mutual(doc_path=wmunit_doc_path, mutual_times=args.mutual_times)
    if args.inject == 1:
        wm_inject_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path,aim_count=args.inject_aim_count)
    filter_flag = False
    if args.verify == 1:
        wm_verify_run_k(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path, filter_flag=False,k=args.verify_k)
    if args.stat == 1:
        stat_result( verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    
    # filter_flag ==True
 
    # wmunit_verify_path = os.path.join(basepath,args.model_name_rllm ,str(args.mutual_times),'wmuint_verify_filter.json')
    # wmunit_stat_path = os.path.join(basepath,args.model_name_rllm ,str(args.mutual_times),'wmuint_stat_filter.json')

    # if args.verify == 1:
    #     wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path, filter_flag=True)
    # if args.stat == 1:
    #     stat_result( verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

## smj

 
# python src/main.py --eval_dataset "msmarco" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 1 
# 
# --inject 1 --verify 1 --stat 1

 
# python src/main.py --eval_dataset "hotpotqa" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 1 --inject 1 --verify 1 --stat 1

#  for model 'llama7b'
 
# python src/main.py --eval_dataset "nfcorpus" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 0 --inject 0  --verify 1 --stat 1 --model_name_rllm 'llama7b'

# 0
# python src/main.py --eval_dataset "trec-covid" --eval_model_code "contriever" --score_function 'cosine'  --mutual_times 10 --doc 0 --inject 0  --verify 1 --stat 1 --model_name_rllm 'llama7b' --gpu_id 3