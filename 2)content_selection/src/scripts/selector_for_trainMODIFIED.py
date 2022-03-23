# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging
from summ import compute_rouge
import json

from pyrouge import Rouge155
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from tqdm import tqdm
import shutil
from multiprocessing import Pool

"""
    This script builds train/val/test datasets for training summarization model on MultiNews. 
    Before using this script, you should build train/val/test datasets for ROUGE Regression first.
    This script ranks all sentences/passages (determined by N_SENTS) according to a metric (e.g., ROUGE Recall or F1). 
    During training Summarization model, you can take the TopK of them by setting args.max_n_passages. 
"""

from data.wiki_parser import WikiParser ##########################################################
import argparse ##################################################################################

def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--project_path", default=None, type=str, required=True)
    parser.add_argument("--dataset_var", default=None, type=str, required=True)
    parser.add_argument("--n_sents", default=1, type=int)
    parser.add_argument("--min_nw_mode", default='sample', type=str)
    parser.add_argument("--selection_metric", default='rouge_2_f1', type=str)
    args = parser.parse_args()
    return args

args = get_args()
DP_PROJ = Path(args.project_path) #UPF CLUSTER

DP_DATA = DP_PROJ / 'data'
DATASET_VARS = ['train', 'val', 'test', 'debug']
MIN_NW_MODES = ['max', 'sample']  # for masked summary file names

DATASET_VAR = args.dataset_var

## Specify the following configs
N_SENTS = args.n_sents  # 1, 4, 8
MIN_NW_MODE = args.min_nw_mode
# N_TOP = 90
## Specify the above configs

if N_SENTS == 1:
    RR_ID_KEY = 'sid'
    RR_PASSAGE_KEY = 'sentence'
else:
    RR_ID_KEY = 'pid'
    RR_PASSAGE_KEY = 'passage'

if MIN_NW_MODE not in MIN_NW_MODES:
    raise ValueError(f'Invalid min_nw_mode: {min_nw_mode}')

######################################################################################################################################################################

DP_MASKED_QUERY = DP_DATA / 'masked_wiki_query' / MIN_NW_MODE
DP_ROUGE_REGRESSION = DP_DATA / 'rouge_regression' / f'rr_{MIN_NW_MODE}_{N_SENTS}'

SELECTION_METRIC = args.selection_metric  # rouge_2_recall, rouge_2_f1

DP_TOP = DP_DATA / 'top_wiki' / f'top_wiki_{MIN_NW_MODE}_{N_SENTS}_{SELECTION_METRIC.split("_")[-1]}'

if not exists(DP_MASKED_QUERY):
    raise ValueError(f'DP_MASKED_QUERY does not exists: {DP_MASKED_QUERY}. MASK WIKI summaries before constructing Top WIKI datasets.')

if not exists(DP_ROUGE_REGRESSION):
    raise ValueError(f'DP_ROUGE_REGRESSION does not exists: {DP_ROUGE_REGRESSION}. Construct RR datasets before constructing Top WIKI datasets.')

if exists(DP_TOP):
    raise ValueError(f'DP_TOP already exists: {DP_TOP}')
os.mkdir(DP_TOP)


def get_cid2query(dataset_var):
    
    #if dataset_var == 'train_debug':  # partial training set for model dev
    #    dataset_var = 'train'
        
    masked_query_fp = DP_MASKED_QUERY / f'{dataset_var}.json'
   
    cid2query = {}
    with open(masked_query_fp) as masked_query_f:
        for line in masked_query_f:
            json_obj = json.loads(line)
            cid = json_obj['cid']
            cid2query[cid] = {
                'original_query': json_obj['raw_query'],
                'masked_query': json_obj['masked_query'],
                'masked_query_with_sub': json_obj['masked_query_with_sub']
            }
    
    return cid2query
    

def _get_cid(json_obj):
    return int(json_obj[RR_ID_KEY].split('_')[0])


def _rank_passage_objs(passage_objs, metric):
    return sorted(passage_objs, key=lambda po: po[metric], reverse=True)


def build(dataset_var):
    
    #if dataset_var not in DATASET_VARS:
    #    raise ValueError(f'Invalid dataset_var: {dataset_var}')

    rr_fp = DP_ROUGE_REGRESSION / f'{dataset_var}.json'
    dump_fp = DP_TOP / f'{dataset_var}.json'
    cid2query = get_cid2query(dataset_var)

    cid = 0
    passages_objs = []
    with open(dump_fp, 'a+') as dump_f:
        with open(rr_fp) as rr_f:
            for line in rr_f:
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                if _cid != cid:
                    ranked_passages_objs = _rank_passage_objs(passages_objs, metric=SELECTION_METRIC)

                    if cid % 500 == 0:
                        logger.info(f'cid: {cid}, #Passages: {len(passages_objs)}')
                    
                    cluster_obj = {
                        'cid': cid,
                        'passages': ranked_passages_objs,
                        **cid2query[cid],  # this include masked and original summary
                    }
                    json_str = json.dumps(cluster_obj, ensure_ascii=False)
                    dump_f.write(f'{json_str}\n')

                    passages_objs = []

                po = {
                    'id': json_obj[RR_ID_KEY],
                    'passage': json_obj[RR_PASSAGE_KEY],
                    'rouge_2_recall': json_obj['rouge_2_recall'],
                    'rouge_2_f1': json_obj['rouge_2_f1'],
                }
                passages_objs.append(po)
                cid = _cid
    
    logger.info(f'Sucessfully dump {dataset_var} set to: {dump_fp}!')


def build_all():
    for dataset_var in DATASET_VARS:
        build(dataset_var)
    

if __name__ == "__main__":
    build(dataset_var=DATASET_VAR)
    #build_all()
