import numpy as np
import json
from collections import defaultdict
import pandas as pd
import itertools
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###############################################
# triplets_path = '../data/triplets/no_overlap_triplets_r4.csv'
# node_path = '../data/graph_node/train_node_freq.json'
triplets_path = '../data/triplets/no_overlap_triplets.csv'
train_node_freq_path = '../data/graph_node/train_node_freq_whole.json'

triplet_df = pd.read_csv(triplets_path, low_memory=False)
positive_doc_df = triplet_df.groupby('qid')['doc+'].apply(list).reset_index(name='positive_docs')
negative_doc_df = triplet_df.groupby('qid')['doc-'].apply(list).reset_index(name='negative_docs')

with open(train_node_freq_path) as f:
    train_node_dict = json.load(f)
print("len of train node dict: {}".format(len(train_node_dict)))
train_graph_path = '../data/graph_node/train_graph_whole.json'
with open(train_graph_path) as f:
    train_graph_dict = json.load(f)

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

# bm25 preselect top20
preselect_path = '../data/preselect/BM25_preselect_100.txt'
test_node_freq_path = '../data/graph_node/test_node_freq_whole.json'

preselect_df = pd.read_csv(preselect_path, sep=" ", names=["qid", "Q0", "docid", "rank", "score", "tag"])
preselect_top20_df = preselect_df.groupby(['qid']).head(20)
test_graph_path = '../data/graph_node/test_graph_whole.json'
with open(test_graph_path) as f:
    test_graph_dict = json.load(f)

with open(test_node_freq_path) as f:
    test_node_dict = json.load(f)

qid_to_bm25_top20_dict = defaultdict(list)
for index, row in preselect_top20_df.iterrows():
    qid = row["qid"]
    docid = row["docid"]
    qid_to_bm25_top20_dict[qid].append(docid)

###############################################
# construct corpus
docid2index = defaultdict()
index2docid = defaultdict()

index = 0
for docid in train_node_dict:
    docid2index[docid] = index
    index2docid[index] = docid
    index += 1
for docid in test_node_dict:
    docid2index[docid] = index
    index2docid[index] = docid
    index += 1

corpus = []
num_train = 0
for i in range(len(train_node_dict)):
    docid = index2docid[i]
    word_list = []
    words = train_node_dict[docid]
    for j in range(len(words)):
        word = words[str(j)]["word"]
        freq = words[str(j)]["freq"]
        for k in range(freq):
            word_list.append(word)
    text = ' '.join(word_list)
    corpus.append(text)
num_train_doc = i + 1
for i in range(len(test_node_dict)):
    index = i + num_train_doc
    docid = index2docid[index]
    word_list = []
    words = test_node_dict[docid]
    for j in range(len(words)):
        word = words[str(j)]["word"]
        freq = words[str(j)]["freq"]
        for k in range(freq):
            word_list.append(word)
    text = ' '.join(word_list)
    corpus.append(text)
print("num of lines in corpus: {}".format(len(corpus)))

###############################################
vectorizer = TfidfVectorizer(stop_words='english')
trsfm = vectorizer.fit_transform(corpus)

posbm25_pair_similarity_score = []
contrary_pair_similarity_score = []

# for qid in qid_list:
for qid in qid_to_pos_doclist_dict.keys():
    pos_doclist = qid_to_pos_doclist_dict[qid]
    bm25_doclist = qid_to_bm25_top20_dict[qid]
    neg_doclist = qid_to_neg_doclist_dict[qid]

    # all_pos_pairs = list(combinations(pos_doclist, 2))
    all_posbm25_pairs = list(itertools.product(pos_doclist, bm25_doclist))
    for posbm25_pair in all_posbm25_pairs:
        pos_id = posbm25_pair[0] + '.txt'
        bm25_id = posbm25_pair[1] + '.txt'
        if pos_id in train_node_dict and bm25_id in test_node_dict:
            if len(train_graph_dict[pos_id])==0:
                print(f'pos id {pos_id} has no edge')
                continue
            elif len(test_graph_dict[bm25_id])==0:
                print(f'bm25 id {bm25_id} has no edge')
                continue
            else:
                index_pos = docid2index[pos_id]
                index_bm25 = docid2index[bm25_id]
                similarity_score = cosine_similarity(trsfm[index_pos], trsfm[index_bm25])
                posbm25_pair_similarity_score.append(similarity_score)
        elif pos_id not in train_graph_dict:
            continue
        else:
            continue

    # all_contrary_pairs = list(itertools.product(pos_doclist, neg_doclist))
    # positive pairs (pos1, pos2)
    all_contrary_pairs = list(combinations(pos_doclist, 2))
    for contrary_pair in all_contrary_pairs:
        pos_id = contrary_pair[0] + '.txt'
        neg_id = contrary_pair[1] + '.txt'
        if pos_id in train_node_dict and neg_id in train_node_dict:
            if len(train_graph_dict[pos_id])==0:
                print(f'pos id {pos_id} has no edge')
                continue
            elif len(train_graph_dict[neg_id])==0:
                print(f'neg id {neg_id} has no edge')
                continue
            else:
                index_pos = docid2index[pos_id]
                index_neg = docid2index[neg_id]
                similarity_score = cosine_similarity(trsfm[index_pos], trsfm[index_neg])
                contrary_pair_similarity_score.append(similarity_score)
        elif pos_id not in train_graph_dict:
            # print(f'pos id {pos_id} is not contained in train_graph_dict!')
            continue
        else:
            # print(f'neg id {neg_id} is not contained in train_graph_dict!')
            continue

print("posbm25_pair_num: ", len(posbm25_pair_similarity_score))
print("average posbm25_pair_similarity_score: ", float(sum(posbm25_pair_similarity_score) / float(len(posbm25_pair_similarity_score))))

print("con_pair_num: ", len(contrary_pair_similarity_score))
print("average contrary_pair_similarity_score: ", float(sum(contrary_pair_similarity_score) / float(len(contrary_pair_similarity_score))))
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
posbm25_pair_similarity_score_arr = np.asarray(posbm25_pair_similarity_score)
contrary_pair_similarity_score_arr = np.asarray(contrary_pair_similarity_score)
# calculate the t test
alpha = 0.05
# t_stat, df, cv, p = independent_ttest(posbm25_pair_similarity_score_arr, contrary_pair_similarity_score_arr, alpha)
t_stat, df, cv, p = independent_ttest(contrary_pair_similarity_score_arr, posbm25_pair_similarity_score_arr, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.8f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
	print('Tfidf tstat: Accept null hypothesis that the means are equal.')
else:
	print('Tfidf tstat: Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
	print('Tfidf p: Accept null hypothesis that the means are equal.')
else:
	print('Tfidf p: Reject the null hypothesis that the means are equal.')