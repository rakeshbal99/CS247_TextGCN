import os
from collections import Counter

import networkx as nx

import itertools
import math
from collections import defaultdict
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np

from utils import print_graph_detail

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_window(content_lst, window_size):
    word_window_freq = defaultdict(int)  # w(i)
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()

        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            if window_size == 10:
                for j in range(0, length - window_size + 1, int(window_size/2)):
                    window = words[j: j + window_size]
                    windows.append(list(set(window)))
            else:
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0., truncate=False):
    if isinstance(content_lst, str):
        content_lst = list(open(content_lst, "r"))
    print("pmi read file len:", len(content_lst))
    
    content_lst = [content.strip() for content in content_lst]
    if truncate:
        content_lst = [' '.join(content.split(' ')[:50]) if len(content.split(' '))>50 else content for content in content_lst]

    pmi_start = time()
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,
                                                                window_size=window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    print("Total number of edges between word:", len(pmi_edge_lst))
    pmi_time = time() - pmi_start
    return pmi_edge_lst, pmi_time


class BuildGraph:
    def __init__(self, dataset, opti_type):
        clean_corpus_path = "data/text_dataset/clean_corpus"
        if opti_type == "truncate":
            self.graph_path = "data/graph_eff"
        elif opti_type == "docu_only":
            self.graph_path = "data/graph_bert"
        else:
            self.graph_path = "data/graph_eff_window"
        self.opti_type = opti_type
        print(self.opti_type)
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        self.word2id = dict()
        self.dataset = dataset
        print(f"\n==> Current:{dataset}<==")

        self.g = nx.Graph()
        self.content = f"{clean_corpus_path}/{dataset}.txt"

        if opti_type == "docu_only":
            self.get_bert_edge()
        else:
            self.get_tfidf_edge()
            self.get_pmi_edge()
        self.save()
        
    def get_bert_edge(self):
        np.random.seed(1234)
        
        start = time()
        
        docs = open(self.content, "r")
        docs = [doc.strip() for doc in docs]
        docs = [' '.join(doc.split(' ')[:50]) if len(doc.split(' '))>50 else doc for doc in docs]
        transform = np.random.rand(384, 5)
        
        doc_emb = np.zeros((len(docs), 384))
        for st in tqdm(range(0, len(docs), 32)):
            if st + 32 > len(docs):
                emb = model.encode(docs[st:])
            else:
                emb = model.encode(docs[st: st+32])
            doc_emb[st: st+len(emb)] = emb
        
        new_doc_emb = np.dot(doc_emb, transform)
        new_doc_emb = np.array([x / math.sqrt(np.dot(x, x)) for x in new_doc_emb])
        edges = np.dot(new_doc_emb, np.transpose(new_doc_emb))
        
        for i in tqdm(range(len(new_doc_emb))):
            for j in range(i, len(new_doc_emb)):
                if edges[i][j] >= 0.9:
                    self.g.add_edge(i, j, weight=edges[i][j])
                    self.g.add_edge(j, i, weight=edges[i][j])
        
        print("bert_graph:", time() - start)
        
        print_graph_detail(self.g)

    def get_pmi_edge(self):
        if self.opti_type == "window":
            pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, window_size=10, threshold=0.0)
        else:
            if self.opti_type == "truncate":
                pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, window_size=20, threshold=0.0, truncate=True)
            else:
                pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, window_size=20, threshold=0.0)
        print("pmi time:", self.pmi_time)

        for edge_item in pmi_edge_lst:
            if edge_item[0] not in self.word2id or edge_item[1] not in self.word2id:
                continue
            word_indx1 = self.node_num + self.word2id[edge_item[0]]
            word_indx2 = self.node_num + self.word2id[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.g.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        print_graph_detail(self.g)

    def get_tfidf_edge(self):
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num + col_ind
                self.g.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst.append(count)

        print_graph_detail(self.g)

    def get_tfidf_vec(self):
        start = time()
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        docs = open(self.content, "r")
        docs = [doc.strip() for doc in docs]
        if self.opti_type == "truncate":
            docs = [' '.join(doc.split(' ')[:50]) if len(doc.split(' '))>50 else doc for doc in docs]
        tfidf_vec = text_tfidf.fit_transform(docs)

        self.tfidf_time = time() - start
        print("tfidf time:", self.tfidf_time)
        print("tfidf_vec shape:", tfidf_vec.shape)
        print("tfidf_vec type:", type(tfidf_vec))

        self.node_num = tfidf_vec.shape[0]

        vocab_lst = text_tfidf["vect"].get_feature_names()
        print("vocab_lst len:", len(vocab_lst))
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind

        self.vocab_lst = vocab_lst

        return tfidf_vec

    def save(self):
        # print("total time:", self.pmi_time + self.tfidf_time)
        nx.write_weighted_edgelist(self.g,
                                   f"{self.graph_path}/{self.dataset}.txt")

        print("\n")


def main():
    for opti_type in ["truncate", "docu_only", "window"]:
        BuildGraph("mr", opti_type)
        BuildGraph("ohsumed", opti_type)
        BuildGraph("R52", opti_type)
        BuildGraph("R8", opti_type)
        BuildGraph("20ng", opti_type)

if __name__ == '__main__':
    main()