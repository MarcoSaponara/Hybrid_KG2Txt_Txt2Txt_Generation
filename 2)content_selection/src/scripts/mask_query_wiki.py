# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import re
from os import listdir
import itertools
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

#from allennlp.predictors.predictor import Predictor

import re ########################################################################################
from data.wiki_parser import WikiParser ##########################################################
import argparse ##################################################################################

def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--project_path", default=None, type=str, required=True)
    parser.add_argument("--dataset_var", default=None, type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()
DP_PROJ = Path(args.project_path) #UPF CLUSTER

DP_DATA = DP_PROJ / 'data'
DP_MASKED_QUERY = DP_DATA / 'masked_wiki_query'

DATASET_VARS = ['train', 'val', 'test']#, 'debug']

WIKI_DATASET_SIZE = {
    'train': None,
    'val': 1598,
    'test': None,
    'debug': 10
}

class QueryMasker:
    def __init__(self, dataset_var, sub_token='[SUBQUERY]'):
      
        self.mask_token = '[MASK]'
        self.dataset_var = dataset_var
        self.sub_token = sub_token
        
        self.wiki_parser = WikiParser(dataset_var=dataset_var)
        
        self.mask_fp = join(DP_MASKED_QUERY, '{}.json'.format(dataset_var))
        self.masked_dicts = self.mask()
    
    def _mask(self, rdf):
      
      s, p, o = rdf.split(' | ')
      s = s.replace('_', ' ')
      p = re.sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', p).lower()
      o = o.replace('_', ' ')
    
      masked_triple = s + ' ' + self.mask_token + ' ' + p + ' ' + self.mask_token + ' ' + o
      return masked_triple.replace("\"", "")
           
    def mask_with_sub(self, masked_rdfs):
        with_sub = ''
        for idx, masked_rdf in enumerate(masked_rdfs):
            
            if idx == 0:
              with_sub += self.sub_token + ' ' + masked_rdf
            else:  
              with_sub += masked_rdf

            if idx < len(masked_rdfs)-1:  # the last token should be appended at after being trimmed (potentially)
                with_sub += ' ' + self.sub_token + ' '
        
        return with_sub
        
    def mask(self):
      
        sample_generator = self.wiki_parser.sample_generator()
        
        outputs = []

        for cid, _, query_rdf, _ in tqdm(sample_generator, total=self.wiki_parser.get_len()):
            masked_rdfs = [self._mask(rdf=rdf) for rdf in query_rdf]
            with_sub = self.mask_with_sub(masked_rdfs = masked_rdfs)
            output = {
                'cid': cid,
                'raw_query': query_rdf,
                'masked_query': masked_rdfs,
                'masked_query_with_sub': with_sub,
            }
            outputs.append(output)

        return outputs 

    def dump_mask(self):
        with open(self.mask_fp, 'a+') as f:
            for md in self.masked_dicts:
                f.write(json.dumps(md, ensure_ascii=False)+'\n')


def mask_and_dump_e2e():
    #for dataset_var in DATASET_VARS:
    q_masker = QueryMasker(dataset_var=args.dataset_var)
    q_masker.dump_mask()


if __name__ == "__main__":
    mask_and_dump_e2e()
