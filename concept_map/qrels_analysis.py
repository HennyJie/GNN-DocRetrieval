import pandas as pd
import numpy as np
from collections import defaultdict
import json
import itertools
from itertools import combinations
from scipy.stats import ttest_ind

#################################################
# node_edge_overlap_functions
def local_id_to_word(docid, words_id_list, node_dict):
    words_list = []
    doc_nodes_dict = node_dict[str(docid)]

    for word_id in words_id_list:
        words_list.append(doc_nodes_dict[str(word_id)])

    return words_list

def local_id_to_word_with_freq(docid, words_id_list, node_dict_with_freq):
    words_freq = {}
    doc_nodes_dict = node_dict_with_freq[str(docid)]

    for word_id in words_id_list:
        word = doc_nodes_dict[str(word_id)]['word']
        freq = doc_nodes_dict[str(word_id)]['freq']
        words_freq[word] = freq

    return words_freq

def local_ids_to_word_tuple(docid, edge_list, node_dict):
    words_tuple_list = []
    doc_nodes_dict = node_dict[str(docid)]

    for edge in edge_list:
        start_node_id = edge[0]
        end_node_id = edge[1]
        words_tuple_list.append(
            (doc_nodes_dict[str(start_node_id)], doc_nodes_dict[str(end_node_id)]))

    return words_tuple_list

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def calculate_node_overlap_perpair(words_list1, words_list2):
    nodes_intersect = intersect(words_list1, words_list2)
    nodes_union = union(words_list1, words_list2)

    nodes_intersect_rate = float(float(
        len(nodes_intersect)) / float((len(nodes_union))))

    return nodes_intersect_rate

def calculate_node_overlap_perpair_with_freq(words_list1, words_list2):
    nodes_intersect = 0

    key_intersect = list(set(words_list1.keys()) & set(words_list2.keys()))
    for key in key_intersect:
        if key != "":
            nodes_intersect += min(words_list1[key], words_list2[key])

    # key_union = list(set(words_list1.keys()) | set(words_list2.keys()))
    # for key in key_union:
    #     nodes_union += sum(words_list1[key], words_list2[key])
    intersect_1 = nodes_intersect / sum(words_list1.values())
    intersect_2 = nodes_intersect / sum(words_list2.values())
    nodes_intersect_rate = (intersect_1 + intersect_2) / 2.0

    return nodes_intersect_rate

def calculate_edge_overlap_perpair(edges_list1, edges_list_2):
    edges_intersect = list(set(map(tuple, edges_list1)).intersection(
        set(map(tuple, edges_list_2))))

    edges_union = list(set(map(tuple, edges_list1)).union(
        set(map(tuple, edges_list_2))))

    edge_intersect_rate = float(float(
        len(edges_intersect)) / float((len(edges_union))))

    return edge_intersect_rate

#################################################
# triplets_path = './data/triplets/no_overlap_triplets_r4.csv'
# train_graph_path = './data/graph_node/train_graph.json'
# train_node_path = './data/graph_node/train_node.json'
# train_node_freq_path = "./data/graph_node/train_node_freq.json"
triplets_path = '../data/triplets/no_overlap_triplets.csv'
train_graph_path = '../data/graph_node/train_graph_whole.json'
train_node_path = '../data/graph_node/train_node_whole.json'
train_node_freq_path = "../data/graph_node/train_node_freq_whole.json"

triplet_df = pd.read_csv(triplets_path, low_memory=False)
positive_doc_df = triplet_df.groupby('qid')['doc+'].apply(list).reset_index(name='positive_docs')
negative_doc_df = triplet_df.groupby('qid')['doc-'].apply(list).reset_index(name='negative_docs')

with open(train_graph_path) as f:
    graph_dict = json.load(f)
print("len of graph dict: {}".format(len(graph_dict)))

with open(train_node_path) as f:
    node_dict = json.load(f)
print("len of node dict: {}".format(len(node_dict)))

with open(train_node_freq_path) as f:
    node_dict_with_freq = json.load(f)
print("len of node freq dict: {}".format(len(node_dict_with_freq)))

qid_to_pos_doclist_dict = defaultdict(list)
qid_to_neg_doclist_dict = defaultdict(list)
for index, row in positive_doc_df.iterrows():
    qid = row['qid']
    doc_list = row['positive_docs']
    for doc in doc_list:
        qid_to_pos_doclist_dict[qid].append(doc)

for index, row in negative_doc_df.iterrows():
    qid = row['qid']
    doc_list = row['negative_docs']
    for doc in doc_list:
        qid_to_neg_doclist_dict[qid].append(doc)

print('len of qid_to_pos_doclist_dict: ', len(qid_to_pos_doclist_dict))
print('len of qid_to_neg_doclist_dict: ', len(qid_to_neg_doclist_dict))

positive_node_intersect = []
positive_node_freq_intersect = []
positive_edge_intersect = []

contrary_node_intersect = []
contrary_node_freq_intersect = []
contrary_edge_intersect = []

# for qid in qid_list:
for qid in qid_to_pos_doclist_dict.keys():
    print(f'qid: {qid}')
    pos_doclist = qid_to_pos_doclist_dict[qid]
    neg_doclist = qid_to_neg_doclist_dict[qid]

    # positive pairs (pos1, pos2)
    all_pos_pairs = list(combinations(pos_doclist, 2))
    print(f'num of pos_pairs: {len(all_pos_pairs)}')

    for pos_pair in all_pos_pairs:
        pos1_id = pos_pair[0] + '.txt'
        pos2_id = pos_pair[1] + '.txt'

        if pos1_id in graph_dict and pos2_id in graph_dict:
            if len(graph_dict[pos1_id])==0:
                print(f'pos id {pos1_id} has no edge')
                continue
            elif len(graph_dict[pos2_id])==0:
                print(f'pos id {pos2_id} has no edge')
                continue
            else:
                pos1_edges = graph_dict[pos1_id]
                pos1_nodes = np.unique(pos1_edges)
                pos1_words = local_id_to_word(pos1_id, pos1_nodes, node_dict)
                pos1_words_freq = local_id_to_word_with_freq(pos1_id, pos1_nodes, node_dict_with_freq)
                pos1_edges_tuple = local_ids_to_word_tuple(pos1_id, pos1_edges, node_dict)

                pos2_edges = graph_dict[pos2_id]
                pos2_nodes = np.unique(pos2_edges)
                pos2_words = local_id_to_word(pos2_id, pos2_nodes, node_dict)
                pos2_words_freq = local_id_to_word_with_freq(pos2_id, pos2_nodes, node_dict_with_freq)
                pos2_edges_tuple = local_ids_to_word_tuple(pos2_id, pos2_edges, node_dict)

                nodes_intersect = calculate_node_overlap_perpair(pos1_words, pos2_words)
                nodes_intersect_with_freq = calculate_node_overlap_perpair_with_freq(pos1_words_freq, pos2_words_freq)
                edge_intersect = calculate_edge_overlap_perpair(pos1_edges_tuple, pos2_edges_tuple)
                positive_node_intersect.append(nodes_intersect)
                positive_node_freq_intersect.append(nodes_intersect_with_freq)
                positive_edge_intersect.append(edge_intersect)
        
        elif pos1_id not in graph_dict:
            continue
        else:
            continue
            

    # contrary pairs (pos, neg)
    all_contrary_pairs = list(itertools.product(pos_doclist, neg_doclist))
    print(f'num of contrary_pairs: {len(all_contrary_pairs)}')

    for contrary_pair in all_contrary_pairs:
        pos_id = contrary_pair[0] + '.txt'
        neg_id = contrary_pair[1] + '.txt'

        if pos_id in graph_dict and neg_id in graph_dict:
            if len(graph_dict[pos_id])==0:
                print(f'pos id {pos_id} has no edge')
                continue
            elif len(graph_dict[neg_id])==0:
                print(f'neg id {neg_id} has no edge')
                continue
            else:
                pos_edges = graph_dict[pos_id]
                pos_nodes = np.unique(pos_edges)
                pos_words = local_id_to_word(pos_id, pos_nodes, node_dict)
                pos_words_freq = local_id_to_word_with_freq(pos_id, pos_nodes, node_dict_with_freq)
                pos_edges_tuple = local_ids_to_word_tuple(pos_id, pos_edges, node_dict)

                neg_edges = graph_dict[neg_id]
                neg_nodes = np.unique(neg_edges)
                neg_words = local_id_to_word(neg_id, neg_nodes, node_dict)
                neg_words_freq = local_id_to_word_with_freq(neg_id, neg_nodes, node_dict_with_freq)
                neg_edges_tuple = local_ids_to_word_tuple(neg_id, neg_edges, node_dict)

                nodes_intersect = calculate_node_overlap_perpair(pos_words, neg_words)
                nodes_intersect_with_freq = calculate_node_overlap_perpair_with_freq(pos_words_freq, neg_words_freq)
                edge_intersect = calculate_edge_overlap_perpair(pos_edges_tuple, neg_edges_tuple)
                contrary_node_intersect.append(nodes_intersect)
                contrary_node_freq_intersect.append(nodes_intersect_with_freq)
                contrary_edge_intersect.append(edge_intersect)

        elif pos_id not in graph_dict:
            continue
        else:
            continue


print("pos_pair_num: ", len(positive_node_intersect))
print("average positive_pair_node_intersect_rate: ", float(sum(positive_node_intersect) / float(len(positive_node_intersect))))
print("average positive_pair_node_freq_intersect_rate: ", float(sum(positive_node_freq_intersect) / float(len(positive_node_freq_intersect))))
print("average positive_pair_edge_intersect_rate: ", float(sum(positive_edge_intersect) / float(len(positive_edge_intersect))))

print("con_pair_num: ", len(contrary_node_intersect))
print("average contrary_pair_node_intersect_rate: ", float(sum(contrary_node_intersect) / float(len(contrary_node_intersect))))
print("average contrary_pair_node_freq_intersect_rate: ", float(sum(contrary_node_freq_intersect) / float(len(contrary_node_freq_intersect))))
print("average contrary_pair_edge_intersect_rate: ", float(sum(contrary_edge_intersect) / float(len(contrary_edge_intersect))))

#################################################
# t-test for independent samples
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p

#################################################
# Node Intersect: two independent samples
positive_node_intersect_arr = np.asarray(positive_node_intersect)
contrary_node_intersect_arr = np.asarray(contrary_node_intersect)
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(positive_node_intersect_arr, contrary_node_intersect_arr, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Node Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Node Intersect: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Node Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Node Intersect: Reject the null hypothesis that the means are equal.')

#################################################
# Node Freq Intersect: two independent samples
positive_node_freq_intersect_arr = np.asarray(positive_node_freq_intersect)
contrary_node_freq_intersect_arr = np.asarray(contrary_node_freq_intersect)
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(positive_node_freq_intersect_arr, contrary_node_freq_intersect, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Node Freq Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Node Freq Intersect: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Node Freq Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Node Freq Intersect: Reject the null hypothesis that the means are equal.')

#################################################
# Edge Intersect: two independent samples
positive_edge_intersect_arr = np.asarray(positive_edge_intersect)
contrary_edge_intersect_arr = np.asarray(contrary_edge_intersect)
# calculate the t test
alpha = 0.05
t_stat, df, cv, p = independent_ttest(positive_edge_intersect_arr, contrary_edge_intersect_arr, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Edge Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Edge Intersect: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Edge Intersect: Accept null hypothesis that the means are equal.')
else:
	print('Edge Intersect: Reject the null hypothesis that the means are equal.')