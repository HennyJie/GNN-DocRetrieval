import pickle
import networkx as nx
import graphviz
from graphviz import Source
from collections import defaultdict
import json
import os

def do_read(pickle_folder_path: str) -> None:
    docs_node_dict_in = "../data/graph_node/train_node.json"
    with open(docs_node_dict_in) as f:
        docs_node_dict = json.load(f)

    docs_node_freq_dict_out = "../data/graph_node/train_node_freq.json"
    docs_node_dict_freq = defaultdict(lambda:defaultdict(dict))

    with open(pickle_path, 'rb') as fopen:
        all_graphs = pickle.load(fopen)

    for G in all_graphs:
        docid = G.graph['docid']
        print(docid)
        edge_list = []
        doc_node_dict = docs_node_dict[docid]
        doc_node_dict_freq = defaultdict()

        for node_id in doc_node_dict:
            node_word = doc_node_dict[node_id]
            doc_node_dict_freq[node_id] = {'word': node_word, 'freq': G.nodes[node_word]['freq']}

        docs_node_dict_freq[docid] = doc_node_dict_freq

    with open(docs_node_freq_dict_out, 'w') as fp:
        json.dump(docs_node_dict_freq, fp)
        print("saved out train node with freq dict!")


if __name__ == '__main__':
    pickle_path = '../data/graph_node/train.win5.pickle.gz'
    do_read(pickle_path)
