import pickle
import networkx as nx
import graphviz
from graphviz import Source
from collections import defaultdict
import json
import os

def do_read(pickle_path: str, out_dot_path: str) -> None:
    docs_graph_dict = defaultdict(list)
    docs_node_dict = defaultdict(dict)

    # docs_graph_dict_out = "../data/graph_node/train_graph_whole.json"
    # docs_node_dict_out = "../data/graph_node/train_node_whole.json"
    docs_graph_dict_out = "../data/graph_node/train_graph.json"
    docs_node_dict_out = "../data/graph_node/train_node.json"

    with open(pickle_path, 'rb') as fopen:
        all_graphs = pickle.load(fopen)

    # pickle_files = os.listdir(pickle_folder_path)
    # for pickle_file in pickle_files:
    #     pickle_path = os.path.join(pickle_folder_path, pickle_file)
    #     with open(pickle_path, 'rb') as fopen:
    #         G = pickle.load(fopen)

    for G in all_graphs:
        print(G.graph['docid'])
        # if G.graph['docid'] == docid_of_interest:
        #     nx.drawing.nx_pydot.write_dot(G, '%s.dot' % (docid_of_interest))
        #     return   # exit
        edge_list = []
        node_dict = defaultdict()
        node_index = 0

        docid = G.graph['docid']
        nodes = list(G.nodes)
        edges = list(G.edges)

        for edge in edges:
            start_node = edge[0]
            end_node = edge[1]
            if start_node == end_node:
                continue
            if start_node not in node_dict:
                node_dict[start_node] = node_index
                node_index += 1
            if end_node not in node_dict:
                node_dict[end_node] = node_index
                node_index += 1

            if([node_dict[start_node], node_dict[end_node]] not in edge_list):
                edge_list.append(
                    [node_dict[start_node], node_dict[end_node]])
            if([node_dict[end_node], node_dict[start_node]] not in edge_list):
                edge_list.append(
                    [node_dict[end_node], node_dict[start_node]])

        edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))

        docs_graph_dict[docid] = edge_list
        docs_node_dict[docid] = {v: k for k, v in node_dict.items()}

        # nx.drawing.nx_pydot.write_dot(
        #     G, out_dot_path+'/%s.dot' % (G.graph['docid']))

    with open(docs_graph_dict_out, 'w') as fp:
        json.dump(docs_graph_dict, fp)
        print("saved out train graph dict!")

    with open(docs_node_dict_out, 'w') as fp:
        json.dump(docs_node_dict, fp)
        print("saved out train node dict!")


if __name__ == '__main__':
    pickle_path = '../data/graph_node/train.win5.pickle.gz'
    out_dot_path = '../dot_train'

    do_read(pickle_path, out_dot_path)
