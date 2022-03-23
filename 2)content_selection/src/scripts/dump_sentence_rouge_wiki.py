# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging

import json
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from pyrouge import Rouge155

import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from summ import compute_rouge

from data.wiki_parser import WikiParser ##########################################################
import argparse ##################################################################################

def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--project_path", default=None, type=str, required=True)
    parser.add_argument("--dataset_var", default=None, type=str, required=True)
    parser.add_argument("--dp_temp", default=None, type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()
DP_PROJ = Path(args.project_path) #UPF CLUSTER
N_PROC = 10

# DP_RAW_MN = DP_DATA / 'raw_mn'
DP_TEMP = DP_PROJ / args.dp_temp
DP_DATA = DP_PROJ / 'data'
DP_SENTENCE_WIKI = DP_DATA / 'sentence_wiki'

DATASET_VARS = ['train', 'val', 'test', 'debug']

WIKI_DATASET_SIZE = {
    'train': None,
    'val': 1598,
    'test': None,
    'debug': 10
}

"""
    Multi-process script for computing and dumping ROUHE scores for sentences from MultiNews or CNN/DailyMail.
"""

def _archived_mp_get_rouge_handler(dp):
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {ROUGE_EVAL_DP} -x'
        
    r = Rouge155(rouge_dir=ROUGE_DP, rouge_args=rouge_args, log_level=logging.WARNING)
    r.system_dir = dp
    r.model_dir = dp

    gen_sys_file_pat = '(\w*).sentence'
    gen_model_file_pat = '#ID#.summary'
    # gen_sys_file_pat = 'sentence'
    # gen_model_file_pat = 'summary'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    return r


def _mp_get_rouge_handler(dp):
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)
    
    r.system_dir = dp
    r.model_dir = dp

    gen_sys_file_pat = '(\w*).sentence'
    gen_model_file_pat = '#ID#.summary'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    return r


def mp_core_doc(arg):
    sid, sentence, summary = arg
    root = DP_TEMP / f'{sid}'
    if not exists(root):
        os.mkdir(root)

    #with open("to_debug.txt", "a+") as fout:
    #    fout.write(sid+"\n")

    sent_fp = root / f'{sid}.sentence'
    summary_fp = root / f'{sid}.summary'
    open(sent_fp, 'a').write(sentence)
    open(summary_fp, 'a').write(summary)
    
    rouge_handler = _mp_get_rouge_handler(dp=root)
    output = rouge_handler.convert_and_evaluate(split_sentences=True)
    res = rouge_handler.output_to_dict(output)

    data = {
        'sid': sid,
        'sentence': sentence,
        'rouge_1_recall': res['rouge_1_recall'],
        'rouge_2_recall': res['rouge_2_recall'],
        'rouge_1_f1': res['rouge_1_f_score'],
        'rouge_2_f1': res['rouge_2_f_score'],
    }
    
    json_str = json.dumps(data, ensure_ascii=False)
    shutil.rmtree(root)
    return json_str


def mp_dump_sentence_wiki(dataset_var, start=0):
    #if dataset_var not in DATASET_VARS:
    #    raise ValueError(f'Illegal {dataset_var}!')

    if not exists(DP_TEMP):
        os.mkdir(DP_TEMP)
    if not exists(DP_SENTENCE_WIKI):
        os.mkdir(DP_SENTENCE_WIKI)
        
    wiki_parser = WikiParser(dataset_var=dataset_var)

    dump_fp = DP_SENTENCE_WIKI / f'{dataset_var}.json'
    print(f'Ready to dump json data to {dump_fp}')

    sample_generator = wiki_parser.sample_generator()
    
    with open(dump_fp, 'a+') as f:
        for cluster_idx, cluster_sentences, summary, query_rdf, query_lex in tqdm(sample_generator, total=wiki_parser.get_len()):
            if cluster_idx < start:
                continue
            cluster_jsons = []
            for doc_idx, doc_sentences in enumerate(cluster_sentences):
                params = [
                    [f'{cluster_idx}_{doc_idx}_{sent_idx}', sent, summary] for sent_idx, sent in enumerate(doc_sentences)
                ]
                p = Pool(N_PROC)
                doc_jsons = p.map(mp_core_doc, params)
                if doc_jsons:
                    cluster_jsons.extend(doc_jsons)
            f.write('\n'.join(cluster_jsons)+'\n')

    os.rmdir(DP_TEMP)



def check_disk(min_disk=20):
    available = float(list(shutil.disk_usage("/"))[-1]) / (1024 ** 3)
    return available > min_disk


def gathered_mp_dump_sentence_wiki(dataset_var, start=0): #, min_disk=20):
    if not exists(DP_TEMP):
        os.mkdir(DP_TEMP)
    if not exists(DP_SENTENCE_WIKI):
        os.mkdir(DP_SENTENCE_WIKI)

    wiki_parser = WikiParser(dataset_var=dataset_var)
    #if dataset_var not in DATASET_VARS:
    #    raise ValueError(f'Illegal {dataset_var}!')

    if start == 0:
        dump_fp = DP_SENTENCE_WIKI / f'{dataset_var}.json'
    else:
        dump_fp = DP_SENTENCE_WIKI / f'{dataset_var}_{start}.json'
    
    print(f'Ready to dump json data to {dump_fp}')

    sample_generator = wiki_parser.sample_generator()

    #p = Pool(N_PROC)
    with open(dump_fp, 'a+') as f:
        for cluster_idx, cluster_sentences, query_rdf, query_lex in tqdm(sample_generator, total=wiki_parser.get_len()):
            #lex = ' '.join(query_lex)
            #print(cluster_idx)
            for lex in query_lex:
                if int(cluster_idx) < start:
                    continue

                #if cluster_idx % 50 == 0 and not check_disk(min_disk=min_disk):
                #    print(f'Disk less than {min_disk}. Break!')
                #    break

                cluster_params = [[f'{cluster_idx}_{doc_idx}_{sent_idx}', sent, lex]
                    for doc_idx, doc_sentences in enumerate(cluster_sentences)
                    for sent_idx, sent in enumerate(doc_sentences)
                ]
                
                p = Pool(N_PROC)
                cluster_jsons = p.map(mp_core_doc, cluster_params, chunksize=1+len(cluster_params)//N_PROC)
                #print(cluster_jsons)
                f.write('\n'.join(cluster_jsons)+'\n')
                p.close()
    #p.join()
    os.rmdir(DP_TEMP)


if __name__ == "__main__":
    #dataset_var = 'train'
    dataset_var = args.dataset_var
    # start
    # val: 1535, 2250, 2870
    # train: 275. 1900, 5700, 6350, 8200, 23760, 24100, 30250, 43000
    start = 0
    end = -1
    #min_disk = 10
    gathered_mp_dump_sentence_wiki(dataset_var=dataset_var, start=start) #, min_disk=min_disk)


