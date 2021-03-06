# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv

import os
import sys
import json
from io import open
from scipy.stats import pearsonr, spearmanr
# import logging
# logger = logging.getLogger(__name__)
from utils.config_loader import logger


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class RrSentenceProcessor(DataProcessor):
    """Processor for the Rouge Regression data set."""
    def get_train_examples(self, data_dir, rouge_c, metric, no_query):
        """
            rouge_coefficient: for calculating labels
        """
        lines = open(os.path.join(data_dir, "train.json")).readlines()
        return self._create_examples(lines, rouge_c=rouge_c, metric=metric, no_query=no_query)

    def get_dev_examples(self, data_dir, rouge_c, metric, no_query):
        lines = open(os.path.join(data_dir, "val.json")).readlines()
        return self._create_examples(lines, rouge_c=rouge_c, metric=metric, no_query=no_query)

    def get_labels(self):
        """See base class."""
        return [None]
    
    def preprocess_json(self, json_obj, rouge_c, metric, no_query):
        """
            rouge_c: ROUGE coefficient; it controls the smoothing effects from rouge_1_recall.
        """
        if no_query:
            text_a = json_obj['sentence'].replace('\n', '')
            text_b = None
        else:
            if type(json_obj['masked_query']) is str:
                text_a = json_obj['masked_query']
            else: 
                assert type(json_obj['masked_query']) is list # masked_query is a word list
                text_a = ' '.join(json_obj['masked_query'])  
            text_b = json_obj['sentence'].replace('\n', '')
        
        if metric == 'rouge_2_recall':
            smooth_metric = 'rouge_1_recall'
        elif metric == 'rouge_2_f1':
            smooth_metric = 'rouge_1_f1'
        else:
            raise ValueError(f'Invalid smooth_metric: {smooth_metric}')

        label = (1 - rouge_c) * float(json_obj[metric]) + rouge_c * float(json_obj[smooth_metric])
        return text_a, text_b, label

    def _create_examples(self, lines, rouge_c, metric, no_query):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            json_obj = json.loads(line.strip('\n'))
            guid = json_obj['sid']
            text_a, text_b, label = self.preprocess_json(json_obj, rouge_c, 
                metric=metric, 
                no_query=no_query)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

'''
REMARK : RrPassageProcessor SHOULD BE MODIFIED
#######################################################################################################################################################################
'''
class RrPassageProcessor(DataProcessor):
    """Processor for the Rouge Regression data set."""
    def get_train_examples(self, data_dir, rouge_c):
        """
            rouge_coefficient: for calculating labels
        """
        lines = open(os.path.join(data_dir, "train.json")).readlines()
        return self._create_examples(lines, rouge_c=rouge_c)

    def get_dev_examples(self, data_dir, rouge_c):
        lines = open(os.path.join(data_dir, "val.json")).readlines()
        return self._create_examples(lines, rouge_c=rouge_c)

    def get_labels(self):
        """See base class."""
        return [None]
    
    def preprocess_json(self, json_obj, rouge_c):
        """
            rouge_c: ROUGE coefficient; it controls the smoothing effects from rouge_1_recall.
        """
        text_a = ' '.join(json_obj['masked_query'])  # masked_query is a word list
        text_b = [sentence.replace('\n', '') for sentence in json_obj['passage']] #??

        label = (1 - rouge_c) * float(json_obj['rouge_2_recall']) + rouge_c * float(json_obj['rouge_1_recall'])
        return text_a, text_b, label

    def _create_examples(self, lines, rouge_c):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            json_obj = json.loads(line.strip('\n'))
            guid = json_obj['pid']
            text_a, text_b, label = self.preprocess_json(json_obj, rouge_c)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
#######################################################################################################################################################################

def convert_examples_to_features(examples, 
                                 max_seq_length,
                                 tokenizer, 
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sub_token='[unused1]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 with_sub=None):
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        if example.text_b:
            if type(example.text_b) == str:
                tokens_b = tokenizer.tokenize(example.text_b)
            else:
                tokens = [tokenizer.tokenize(sentence)[:50] for sentence in example.text_b]
                import itertools
                tokens_b = list(itertools.chain(*tokens))

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            if with_sub: # CLS, SUBQUERY, SEP, SEP
                special_tokens_count = 4
            else:  # CLS, SEP, SEP
                special_tokens_count = 3

            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length-special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        if with_sub and tokens_b:  # if there's no tokens_b, tokens_a is cand sentence
            tokens_a = [sub_token if token == '[SUBQUERY]' else token for token in tokens_a]
            tokens_a += [sub_token]
        
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        tokens = [cls_token] + tokens
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = float(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "rr":
        return pearson_and_spearman(preds, labels)
    elif task_name == "rr-p":
        return pearson_and_spearman(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "rr": RrSentenceProcessor,
    "rr-p": RrPassageProcessor,
}

GLUE_TASKS_NUM_LABELS = {
    "rr": 1,
    "rr-p": 1,
}
