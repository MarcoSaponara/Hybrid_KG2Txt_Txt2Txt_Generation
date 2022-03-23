# -*- coding: utf-8 -*-
import sys
from os.path import isfile, isdir, join, dirname, abspath, exists
from os import listdir

sys_path = dirname(dirname(abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

parent_sys_path = dirname(sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

parent_sys_path = dirname(parent_sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config
from utils.config_loader import config_model, path_parser
from data.dataset_parser import dataset_parser ########################################################################################################################
import data.data_tools as data_tools ##################################################################################################################################
from bert_rr.bert_input import build_bert_sentence_x

import io
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
    Function: data pipe for the MultiNews test set
    Method: use a placeholder query, or an empty query, to rank sentences in the test set
"""

def load_retrieved_sentences(retrieved_dp, cid):
    """
        This func was mainly copied from ir.ir_tools but has been changed as follows:
            Origin: 
                processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=True, rm_stop=True, stem=True)
                       for ss in original_sents]
            Now:
                processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)
                       for ss in original_sents]
            
            Reason of change: you can filter DIALOG sentences during IR. 
    :param retrieved_dp:
    :param cid:
    :return:
    """
    if not exists(retrieved_dp):
        raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()

    original_sents = [ll.rstrip('\n').split('\t')[-1] for ll in content]

    processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)
                       for ss in original_sents]

    return [original_sents], [processed_sents]  # for compatibility of document organization for similarity calculation


def get_test_cc_ids(rel_scores_dp):
    cc_ids = [fn for fn in listdir(rel_scores_dp) if isfile(join(rel_scores_dp, fn))]
    return cc_ids


def get_query(n_summary_sents):
    item = ['[MASK]', '.']
    query = ' '.join(item * n_summary_sents)
    return query


class ClusterDataset(Dataset):
    def __init__(self, cid, retrieve_dp, n_summary_sents, no_query, transform=None):
        super(ClusterDataset, self).__init__()
        original_sents, _ = load_retrieved_sentences(retrieved_dp=retrieve_dp, cid=cid)
        self.sentences = [sent.replace('\n', '') for sent in original_sents[0]]

        self.n_summary_sents = n_summary_sents
        self.query = get_query(n_summary_sents) if not no_query else None
        self.yy = 0.0

        self.transform = transform

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def _vec_label(yy):
        if yy == '-1.0':
            yy = 0.0
        return np.array([yy], dtype=np.float32)

    def __getitem__(self, index):
        """∂
            get an item from self.doc_ids.
            return a sample: (xx, yy)
        """
        # if index == 0:
            # print(f'query: {self.query}')
        # build xx
        xx = build_bert_sentence_x(self.query, sentence=self.sentences[index])

        # build yy
        yy = self._vec_label(self.yy)

        sample = {
            **xx,
            'yy': yy,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ClusterDataLoader(DataLoader):
    def __init__(self, cid, retrieve_dp, n_summary_sents, no_query, transform=data_tools.ToTensor()):
        dataset = ClusterDataset(cid, retrieve_dp=retrieve_dp, n_summary_sents=n_summary_sents, no_query=no_query)
        self.transform = transform
        self.cid = cid

        super(ClusterDataLoader, self).__init__(dataset=dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=10,  # 3
                                                drop_last=False)

    def _generator(self, super_iter):
        while True:
            batch = next(super_iter)
            batch = self.transform(batch)
            yield batch

    def __iter__(self):
        super_iter = super(ClusterDataLoader, self).__iter__()
        return self._generator(super_iter)


class QSDataLoader:
    """
        iter over all clusters.
        each cluster is handled with a separate data loader.
        tokenize_narr: whether tokenize query into sentences.
    """

    def __init__(self, retrieve_dp, n_summary_sents, no_query):
        cids = get_test_cc_ids(rel_scores_dp=retrieve_dp)

        self.loader_init_params = []
        for cid in cids:
            self.loader_init_params.append({
                'cid': cid,
                'retrieve_dp': retrieve_dp,
                'n_summary_sents': n_summary_sents,
                'no_query': no_query,
            })

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            yield c_loader

    def __iter__(self):
        return self._loader_generator()
