# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 16:04:02 2021
@author: marco
"""

from corpusReader.benchmark_reader import Benchmark, select_files
from datasets import load_dataset
import re
import numpy as np
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--split", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    args = parser.parse_args()
    return args

def get_webnlg_data(data_path, split):
  if split =='test':
    with open(data_path+'/'+split+'.json') as f:
        test_data = json.load(f)
    f.close()
    return test_data["entries"]
  else:
    b = Benchmark()
    files = select_files(data_path + '/' + split)
    b.fill_benchmark(files)
    return b.entries

args = get_args()
data_path = args.data_path
split = args.split
output_dir = args.output_dir

rdf_data = get_webnlg_data(data_path, split)

wikipedia = load_dataset("wikipedia", '20200501.en', split='train')

ARTICLE_SEP = '\n|||||\n'

l = len(rdf_data)

keywords_list = []

for entry in rdf_data:
    if split=='test':
        flat_tripleset = '<br>'.join(entry['modifiedtripleset']['mtriple']) 
        relations = set(flat_tripleset.split(' | ')[1::2])
        keywords = set(re.split(' \| |<br>', flat_tripleset)) - relations
    else:
        keywords = set(re.split(' \| |<br>', entry.flat_tripleset())) - entry.relations()
    keywords_list.append(keywords)

documents = len(rdf_data)*[None]

for idx, keywords in enumerate(keywords_list):
  
    keywords = [keyword.replace('_', ' ') for keyword in keywords]

    mask = np.isin(wikipedia['title'], keywords)

    articles = wikipedia[np.nonzero(mask)[0]]['text']

    #preprocessing articles
    proc_articles = []
    for article in articles:
      article = article.split('References')
      if len(article)<=2:
        article = article[0]
      else:
        article = 'References'.join(article[:-1])
      article=article.strip()
      if article!='':
         proc_articles.append(article)
    
    if len(proc_articles)>0:  
        documents[idx] = 'ARTICLE_SEP'.join(proc_articles)

dataset = []

for i in range(l):
  entry = rdf_data[i]

  if documents[i] is not None:
    if split=='test':
        if 'lex' in entry.keys():
            dataset.append({'document': documents[i],
                          'query_rdf': entry['modifiedtripleset']['mtriple'],
                          'query_lex': [lex['#text'] for lex in entry['lex']],
                          }) 
        else:
            dataset.append({'document': documents[i],
                          'query_rdf': entry['modifiedtripleset']['mtriple'],
                          'query_lex': None,
                          })             
    else:
        dataset.append({'document': documents[i],
                      'query_rdf': entry.list_triples(),
                      'query_lex': [lex.lex for lex in entry.lexs],
                      })

output_path = output_dir+'/'+split+'.json'
with open(output_path, 'w+') as f:
    json.dump(dataset, f)
f.close()
