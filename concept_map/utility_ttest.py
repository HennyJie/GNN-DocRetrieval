import pandas as pd
import numpy as np
from collections import defaultdict
import json
import itertools
from itertools import combinations
from scipy.stats import ttest_ind

#################################################
# node_edge_overlap_functions
def local_id_to_words_list(docid, words_id_list, node_dict):
    words_list = []
    doc_nodes_dict = node_dict[str(docid)]
    for word_id in words_id_list:
        words_list.append(doc_nodes_dict[str(word_id)])
    return words_list

def local_id_to_words_freq(docid, words_id_list, node_dict_with_freq):
    words_freq = {}
    doc_nodes_dict = node_dict_with_freq[str(docid)]
    for word_id in words_id_list:
        word = doc_nodes_dict[str(word_id)]['word']
        freq = doc_nodes_dict[str(word_id)]['freq']
        words_freq[word] = freq
    return words_freq

def local_id_to_words_tuple(docid, edge_list, node_dict):
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

def node_overlap(words_list1, words_list2):
    nodes_intersect = intersect(words_list1, words_list2)
    nodes_union = union(words_list1, words_list2)
    nodes_intersect_rate = float( float(len(nodes_intersect)) / float(len(nodes_union)) )
    return nodes_intersect_rate

def node_freq_overlap(words_freq1, words_freq2):
    nodes_intersect = 0
    key_intersect = list( set(words_freq1.keys()) & set(words_freq2.keys()) )
    for key in key_intersect:
        if key != "":
            nodes_intersect += min(words_freq1[key], words_freq2[key])
    intersect_1 = nodes_intersect / sum(words_freq1.values())
    intersect_2 = nodes_intersect / sum(words_freq2.values())
    nodes_intersect_rate = (intersect_1 + intersect_2) / 2.0
    return nodes_intersect_rate

def edge_overlap(edges_list1, edges_list2):
    edges_intersect = list( set( map(tuple, edges_list1) ).intersection(set( map(tuple, edges_list2) )) )
    edges_union = list( set(map(tuple, edges_list1) ).union(set( map(tuple, edges_list2) )) )
    edge_intersect_rate = float( float(len(edges_intersect)) / float(len(edges_union)) ) 
    return edge_intersect_rate

#################################################
# qrels positive / negative
triplets_path = '../data/triplets/no_overlap_triplets.csv'
train_graph_path = '../data/graph_node/train_graph_whole.json'
train_node_path = '../data/graph_node/train_node_whole.json'
train_node_freq_path = '../data/graph_node/train_node_freq_whole.json'

triplet_df = pd.read_csv(triplets_path, low_memory=False)
positive_doc_df = triplet_df.groupby('qid')['doc+'].apply(list).reset_index(name='positive_docs')
negative_doc_df = triplet_df.groupby('qid')['doc-'].apply(list).reset_index(name='negative_docs')

with open(train_graph_path) as f:
    train_graph_dict = json.load(f)
with open(train_node_path) as f:
    train_node_dict = json.load(f)
with open(train_node_freq_path) as f:
    train_node_freq_dict = json.load(f)

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

# bm25 preselect top20
preselect_path = '../data/preselect/BM25_preselect_100.txt'
test_graph_path = '../data/graph_node/test_graph_whole.json'
test_node_path = '../data/graph_node/test_node_whole.json'
test_node_freq_path = '../data/graph_node/test_node_freq_whole.json'

preselect_df = pd.read_csv(preselect_path, sep=" ", names=["qid", "Q0", "docid", "rank", "score", "tag"])
preselect_top20_df = preselect_df.groupby(['qid']).head(20)

with open(test_graph_path) as f:
    test_graph_dict = json.load(f)
with open(test_node_path) as f:
    test_node_dict = json.load(f)
with open(test_node_freq_path) as f:
    test_node_freq_dict = json.load(f)

qid_to_bm25_top20_dict = defaultdict(list)
for index, row in preselect_top20_df.iterrows():
    qid = row["qid"]
    docid = row["docid"]
    qid_to_bm25_top20_dict[qid].append(docid)

#################################################
# start comparison
posbm25_node_intersect = []
posbm25_node_freq_intersect = []
posbm25_edge_intersect = []
contrary_node_intersect = []
contrary_node_freq_intersect = []
contrary_edge_intersect = []

for qid in qid_to_pos_doclist_dict.keys():
    print(f'qid: {qid}')
    pos_doclist = qid_to_pos_doclist_dict[qid]
    bm25_doclist = qid_to_bm25_top20_dict[qid]
    neg_doclist = qid_to_neg_doclist_dict[qid]

    # bm25pos
    all_posbm25_pairs = list(itertools.product(pos_doclist, bm25_doclist))
    for posbm25_pair in all_posbm25_pairs:
        pos_id = posbm25_pair[0] + '.txt'
        bm25_id = posbm25_pair[1] + '.txt'
        if pos_id in train_graph_dict and bm25_id in test_graph_dict:
            if len(train_graph_dict[pos_id])==0:
                print(f'pos id {pos_id} has no edge')
                continue
            elif len(test_graph_dict[bm25_id])==0:
                print(f'bm25 id {bm25_id} has no edge')
                continue
            else:
                pos_edges = train_graph_dict[pos_id]
                pos_nodes = np.unique(pos_edges)
                pos_words = local_id_to_words_list(pos_id, pos_nodes, train_node_dict)
                pos_words_freq = local_id_to_words_freq(pos_id, pos_nodes, train_node_freq_dict)
                pos_edges_tuple = local_id_to_words_tuple(pos_id, pos_edges, train_node_dict)

                bm25_edges = test_graph_dict[bm25_id]
                bm25_nodes = np.unique(bm25_edges)
                bm25_words = local_id_to_words_list(bm25_id, bm25_nodes, test_node_dict)
                bm25_words_freq = local_id_to_words_freq(bm25_id, bm25_nodes, test_node_freq_dict)
                bm25_edges_tuple = local_id_to_words_tuple(bm25_id, bm25_edges, test_node_dict)

                nodes_intersect = node_overlap(pos_words, bm25_words)
                nodes_freq_intersect = node_freq_overlap(pos_words_freq, bm25_words_freq)
                edge_intersect = edge_overlap(pos_edges_tuple, bm25_edges_tuple)
                posbm25_node_intersect.append(nodes_intersect)
                posbm25_node_freq_intersect.append(nodes_freq_intersect)
                posbm25_edge_intersect.append(edge_intersect)
        
        elif pos_id not in train_graph_dict:
            continue
        else:
            continue
            

    # contrary pairs (pos, neg)
    all_contrary_pairs = list(combinations(pos_doclist, 2))
    for contrary_pair in all_contrary_pairs:
        pos_id = contrary_pair[0] + '.txt'
        neg_id = contrary_pair[1] + '.txt'
        if pos_id in train_graph_dict and neg_id in train_graph_dict:
            if len(train_graph_dict[pos_id])==0:
                print(f'pos id {pos_id} has no edge')
                continue
            elif len(train_graph_dict[neg_id])==0:
                print(f'neg id {neg_id} has no edge')
                continue
            else:
                pos_edges = train_graph_dict[pos_id]
                pos_nodes = np.unique(pos_edges)
                pos_words = local_id_to_words_list(pos_id, pos_nodes, train_node_dict)
                pos_words_freq = local_id_to_words_freq(pos_id, pos_nodes, train_node_freq_dict)
                pos_edges_tuple = local_id_to_words_tuple(pos_id, pos_edges, train_node_dict)

                neg_edges = train_graph_dict[neg_id]
                neg_nodes = np.unique(neg_edges)
                neg_words = local_id_to_words_list(neg_id, neg_nodes, train_node_dict)
                neg_words_freq = local_id_to_words_freq(neg_id, neg_nodes, train_node_freq_dict)
                neg_edges_tuple = local_id_to_words_tuple(neg_id, neg_edges, train_node_dict)

                nodes_intersect = node_overlap(pos_words, neg_words)
                nodes_freq_intersect = node_freq_overlap(pos_words_freq, neg_words_freq)
                edge_intersect = edge_overlap(pos_edges_tuple, neg_edges_tuple)
                contrary_node_intersect.append(nodes_intersect)
                contrary_node_freq_intersect.append(nodes_freq_intersect)
                contrary_edge_intersect.append(edge_intersect)

        elif pos_id not in train_graph_dict:
            continue
        else:
            continue

#################################################
# results
print("posbm25_pair_num: ", len(posbm25_node_intersect))
print("average posbm25_pair_node_intersect_rate: ", float(sum(posbm25_node_intersect) / float(len(posbm25_node_intersect))))
print("average posbm25_pair_node_freq_intersect_rate: ", float(sum(posbm25_node_freq_intersect) / float(len(posbm25_node_freq_intersect))))
print("average posbm25_pair_edge_intersect_rate: ", float(sum(posbm25_edge_intersect) / float(len(posbm25_edge_intersect))))

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
# Node Intersect
posbm25_node_intersect_arr = np.asarray(posbm25_node_intersect)
contrary_node_intersect_arr = np.asarray(contrary_node_intersect)
# calculate the t test
alpha = 0.05
# t_stat, df, cv, p = independent_ttest(posbm25_node_intersect_arr, contrary_node_intersect_arr, alpha)
t_stat, df, cv, p = independent_ttest(contrary_node_intersect_arr, posbm25_node_intersect_arr, alpha)

print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Node Intersect tstat: Accept null hypothesis that the means are equal.')
else:
	print('Node Intersect tstat: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Node Intersect p: Accept null hypothesis that the means are equal.')
else:
	print('Node Intersect p: Reject the null hypothesis that the means are equal.')

#################################################
# Node Freq Intersect
posbm25_node_freq_intersect_arr = np.asarray(posbm25_node_freq_intersect)
contrary_node_freq_intersect_arr = np.asarray(contrary_node_freq_intersect)
# calculate the t test
alpha = 0.05
# t_stat, df, cv, p = independent_ttest(posbm25_node_freq_intersect_arr, contrary_node_freq_intersect_arr, alpha)
t_stat, df, cv, p = independent_ttest(contrary_node_freq_intersect_arr, posbm25_node_freq_intersect_arr, alpha)

print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Node Freq Intersect tstat: Accept null hypothesis that the means are equal.')
else:
	print('Node Freq Intersect tstat: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Node Freq Intersect p: Accept null hypothesis that the means are equal.')
else:
	print('Node Freq Intersect p: Reject the null hypothesis that the means are equal.')

#################################################
# Edge Intersect
posbm25_edge_intersect_arr = np.asarray(posbm25_edge_intersect)
contrary_edge_intersect_arr = np.asarray(contrary_edge_intersect)
# calculate the t test
alpha = 0.05
# t_stat, df, cv, p = independent_ttest(posbm25_edge_intersect_arr, contrary_edge_intersect_arr, alpha)
t_stat, df, cv, p = independent_ttest(contrary_edge_intersect_arr, posbm25_edge_intersect_arr, alpha)

print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Edge Intersect tstat: Accept null hypothesis that the means are equal.')
else:
	print('Edge Intersect tstat: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Edge Intersect p: Accept null hypothesis that the means are equal.')
else:
	print('Edge Intersect p: Reject the null hypothesis that the means are equal.')