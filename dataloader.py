from pathlib import Path
import collections
import numpy as np
import json
from torch_geometric.data import Data, DataLoader
import torch
import pandas as pd

data_folder = '../DATA_Standardized'
preselect_path = 'BM25_preselect_100.txt'
triplet_path = 'no_overlap_triplets.csv'
train_graph_dict_path = 'train_graph_whole.json'
train_node_emb_path = 'train_node_whole_emb.npy'
test_graph_dict_path = 'test_graph_whole.json'
test_node_emb_path = 'test_node_whole_emb.npy'


def load_nodes_features(edges_list, nodes_emb):
    nodes_feature = []
    nodes_degree = collections.Counter(edges_list[0])
    # this iteration in counter can preserve the origin order (1....n)
    for node_id in nodes_degree:
        node_feature = np.array(nodes_emb[node_id])
        nodes_feature.append(node_feature)

    nodes_feature = np.array(nodes_feature)
    return nodes_feature


def load_graph_dict(graph_dict_path, node_emb_path):
    with open(Path(data_folder) / Path(graph_dict_path)) as f:
        graph_dict = json.load(f)

    node_emb_dict = dict(np.load(Path(data_folder) / Path(node_emb_path), allow_pickle=True).item())

    # convert list to ndarray, then to torch.Tensor
    for k, v in graph_dict.items():
        if (len(v) == 0):
            continue
        tmp = np.array(v)
        edges_list = np.transpose(tmp)
        nodes_emb = node_emb_dict[k]
        
        nodes_feature = load_nodes_features(edges_list, nodes_emb)

        graph_dict[k] = Data(x=torch.tensor(nodes_feature, dtype=torch.float), edge_index=torch.tensor(
            edges_list, dtype=torch.long))

    return graph_dict


def get_train_loader(folder_path, query_embedding, batch_size):
    train_graph_dict = load_graph_dict(train_graph_dict_path, train_node_emb_path)
    triplets_df = pd.read_csv(Path(folder_path) / Path(triplet_path))
    # for train dataset, add the triplets to the list
    train_dataset = []
    for row_index, row in triplets_df.iterrows():
        qid = row['qid']
        positive_docid = row['doc+']+'.txt'
        negative_docid = row['doc-']+'.txt'
        if positive_docid in train_graph_dict and negative_docid in train_graph_dict:
            if isinstance(train_graph_dict[positive_docid], list):
                continue
            if isinstance(train_graph_dict[negative_docid], list):
                continue
            train_dataset.append((qid, query_embedding[str(qid)], positive_docid, train_graph_dict[positive_docid], 
                                negative_docid, train_graph_dict[negative_docid]))

    print("number of triplets in train_dataset: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def get_test_loader(folder_path, query_embedding):
    test_graph_dict = load_graph_dict(test_graph_dict_path, test_node_emb_path)
    preselect_df = pd.read_csv(Path(folder_path) / Path(preselect_path), sep=" ", names=["qid", "Q0", "docid", "rank", "score", "tag"])
    # for test dataset, only query-document pairs to the list
    test_dataset = []
    for row_index, row in preselect_df.iterrows():
        qid = row['qid']
        docid = row['docid']+'.txt'
        if docid in test_graph_dict:
            if isinstance(test_graph_dict[docid], list):
                continue
            test_dataset.append((qid, query_embedding[str(qid)], docid, test_graph_dict[docid]))

    print("number of q-doc pairs in test_dataset: {}".format(len(test_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=1)
    return test_loader