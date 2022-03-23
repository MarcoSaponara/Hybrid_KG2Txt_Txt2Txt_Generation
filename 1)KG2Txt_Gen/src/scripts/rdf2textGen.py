# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:47:12 2021

@author: marco
"""

import json, argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_path", default=None, type=str, required=True)
    parser.add_argument("--input_data_path", default=None, type=str, required=True)
    parser.add_argument("--output_data_path", default=None, type=str, required=True)
    args = parser.parse_args()
    return args

def build_input(rdfs):
  input = ''

  for rdf in rdfs:
    s, p, o = rdf.split(' | ')
    proc_rdf = '__subject__ '+s+' __predicate__ '+o+' __object__ '
    input+=proc_rdf
  
  return input

args = get_args()

tokenizer = T5Tokenizer.from_pretrained(args.model_path)
model = T5ForConditionalGeneration.from_pretrained(args.model_path)

with open(args.input_data_path, 'r') as f:
  data = json.load(f)
f.close()

results =[]
for entry in data:
  input_text = build_input(entry['query_rdf'])
  input_ids = tokenizer(input_text, 
                        max_length=1024,
                        truncation=True,
                        return_tensors='pt').input_ids
  
  result = {}
  result['rdf'] = entry['query_rdf']
  outputs = model.generate(input_ids,       
                          max_length=512,
                          early_stopping=True)
  result['text'] = tokenizer.decode(outputs[0],  
                                    skip_special_tokens=True)
  results.append(result)
  
with open(args.output_data_path, 'w') as f:
    json.dump(results, f)
f.close()