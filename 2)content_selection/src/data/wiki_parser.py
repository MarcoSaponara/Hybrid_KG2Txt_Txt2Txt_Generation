# -*- coding: utf-8 -*-
import io
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import re
from os import listdir
from os.path import join, isfile, isdir
import itertools

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta, config_model
import utils.tools as tools

import data.clip_and_mask as cm
import data.clip_and_mask_sl as cm_sl

import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer

from tqdm import tqdm

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

import json ################################################################

"""
    This class provide following information extraction functions:
    (1) get_doc:
        Get an article from file.
        Return natural paragraphs/subtopic tiles/whole artilce.
    (2) doc2sents:
        Based on func:get_doc, get sentences from an article.
        Optionally, sentences can be organized by paragraphs.
    (3) cid2sents:
        Based on func:doc2sents, get sentences from a cluster.
    This class also provide following parsing functions:
    (1) parse_doc2paras:
        Based on func:get_doc, parse a doc => a dict with keys: ('paras',
                                                                 'article_mask')
    (2) parse_doc2sents:
        Based on func:doc2sents,
"""

WIKI_DATASET_SIZE = {
    'train': None,
    'val': None,
    'test': None,
    'debug': 10
}

class WikiParser():
    def __init__(self, dataset_var):
        # self.ARTICLE_SEP = 'story_separator_special_tag'
        # self.SENTENCE_SEP = ' ' * 5
        self.ARTICLE_SEP = '\n|||||\n'
        self.dataset_var = dataset_var
        data_fp = path_parser.wiki / f'{dataset_var}.json'
        
        with open(data_fp) as f:
            self.data = json.load(f)
        
        if type(self.data) is list:
            self.index = range(len(self.data))
        else:
            self.index = list(self.data['document'].keys())

    def get_len(self):
        if type(self.data) is list:
            return len(self.data)
        else:
            return len(self.data['document'])
            
    def get_articles(self, idx):
        if type(self.data) is list:
            return self.data[int(idx)]['document'].split(self.ARTICLE_SEP)
        else:
            return self.data['document'][str(idx)].split(self.ARTICLE_SEP)
    
    def get_sentences(self, article): 
        return nltk.tokenize.sent_tokenize(article)
    '''
    def get_summary(self, idx):
        if type(self.data) is list:
            return self.data[int(idx)]['summary']
        else:
            return self.data['summary'][str(idx)]
    '''
    def get_query_rdf(self, idx):
        if type(self.data) is list:
            return self.data[int(idx)]['query_rdf']
        else:
            return self.data['query_rdf'][str(idx)]
    
    def get_query_lex(self, idx):
        if type(self.data) is list:
            return self.data[int(idx)]['query_lex']
        else:
            return self.data['query_lex'][str(idx)]
    
    def sample_generator(self):
        """
            Generate a sample:
                cluster_idx, sentences (2D, organized via aticles), summary
        """
        for idx in self.index:
            articles = self.get_articles(idx = idx)
            sentences = [self.get_sentences(article=article) for article in articles]
            #summary = self.get_summary(idx = idx)
            query_rdf = self.get_query_rdf(idx = idx)
            query_lex = self.get_query_lex(idx = idx)
            yield (idx, sentences, query_rdf, query_lex)
