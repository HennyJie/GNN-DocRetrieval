import json
import math
from collections import defaultdict

def produce_idf_on_files(files: list) -> tuple:
    idf_dict = defaultdict(int)
    total_doc_cnt = 0
    for fname in files:
        with open(fname, 'r') as fopen:
            docs = json.load(fopen)    # 2D list
            for docid, nodes in docs.items():
                total_doc_cnt += 1
                for nid, node_dict in nodes.items():
                    node = node_dict['word']
                    idf_dict[node] += 1
    idf_dict = {k: math.log2(total_doc_cnt/(1+v))+1 for k, v in idf_dict.items()}   # inverse smooth
    print('[produce_idf] total doc cnt = %d' % (total_doc_cnt))
    return idf_dict, total_doc_cnt


def produce_tfidf(input_path: str, output_path: str, corpus_list: list):
    idf_dict, total_doc_cnt = produce_idf_on_files(corpus_list)
    with open(input_path, 'r') as fopen:
        docs = json.load(fopen)    # 2D list
        for docid, nodes in docs.items():
            for nid, node_dict in nodes.items():
                node = node_dict['word']
                tf = node_dict['freq']
                idf = idf_dict.get(node, math.log2(total_doc_cnt/1)+1)
                node_dict['idf'] = idf
                node_dict['tf-idf'] = tf * idf
    with open(output_path, 'w') as fwrite:
        json.dump(docs, fwrite)
    

if __name__ == '__main__':
    train_path = '../data/graph_node/train_node_freq.json'
    test_path = '../data/graph_node/test_node_freq.json'
    corpus_files = [train_path, test_path]
    train_out_path = '../data/graph_node/train_node_freq_tfidf.json'
    produce_tfidf(train_path, train_out_path, corpus_files)
    test_out_path = '../data/graph_node/test_node_freq_tfidf.json'
    produce_tfidf(test_path, test_out_path, corpus_files)
