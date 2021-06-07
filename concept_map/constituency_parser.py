# Use Constituency Parser and other essential tools to generate Phrase Graph
import json
import pickle
import tqdm
import networkx as nx
from nltk.tree import Tree
from tree_api import extract_NP_VP_from_constituency_parse


def is_within_window(offsets_i: list, offsets_j: list, window_size: int) -> bool:
    for (s_i, e_i) in offsets_i:
        for (s_j, e_j) in offsets_j:
            if abs(s_i - e_j) <= window_size or abs(s_j - e_i) <= window_size:
                return True
    return False


def produce_phrase_graphs_per_doc(doc_dict: dict, window_size: int) -> nx.Graph:
    phrases_doc = []
    lemmas_doc = []
    pos_tags_doc = []
    len_sents_doc = []
    G = nx.Graph()
    G.graph['docid'] = doc_dict['id']   # TODO: this is only for COVID IR corpus
    for sent_dict in doc_dict['sentences']:
        len_sents_doc.append(len(sent_dict['lemmas']))
        lemmas_doc.append(sent_dict['lemmas'])
        parse_tree = Tree.fromstring(sent_dict['parse_tree'])
        pos_tags_doc.append([_[1] for _ in parse_tree.pos()])
        phrases_sent = extract_NP_VP_from_constituency_parse(parse_tree)
        phrases_doc.append(phrases_sent)

    # using coref for pronoun
    coref_mapping = {}
    for coref_cluster in doc_dict['coref_clusters']:
        hit_coref_index = []
        cluster_mention = None   # the mention either from exisiting phrase, or first mention in chain
        for (sent_idx, s_tidx, e_tidx) in coref_cluster:
            if (s_tidx, e_tidx) in phrases_doc[sent_idx]:
                # only replace pronoun
                if e_tidx - s_tidx == 1 and pos_tags_doc[sent_idx][s_tidx] == 'PRP':
                    hit_coref_index.append((sent_idx, s_tidx, e_tidx))
                else:
                    cluster_mention = ' '.join(lemmas_doc[sent_idx][s_tidx:e_tidx])
        if cluster_mention is None:
            # using first mention in chain
            sent_idx, s_tidx, e_tidx = coref_cluster[0]
            cluster_mention = ' '.join(lemmas_doc[sent_idx][s_tidx:e_tidx])
        if len(hit_coref_index) > 0:
            for k_idx in hit_coref_index:
                coref_mapping[k_idx] = cluster_mention
    # Add nodes
    offset = 0
    for sent_idx, phrases_sent in enumerate(phrases_doc):
        for (s_tidx, e_tidx) in phrases_sent:
            if (sent_idx, s_tidx, e_tidx) in coref_mapping:
                mention = coref_mapping[(sent_idx, s_tidx, e_tidx)]
            else:
                mention = ' '.join(lemmas_doc[sent_idx][s_tidx:e_tidx])
            if mention not in G:
                G.add_node(mention, freq=1, offsets=[(offset+s_tidx, offset+e_tidx)])
            else:
                G.nodes[mention]['freq'] += 1
                G.nodes[mention]['offsets'].append((offset+s_tidx, offset+e_tidx))
        offset += len_sents_doc[sent_idx]
    # Add edges
    for i in range(G.number_of_nodes()):
        for j in range(i+1, G.number_of_nodes()):
            node_i = list(G)[i]
            node_j = list(G)[j]
            if is_within_window(G.nodes[node_i]['offsets'], G.nodes[node_j]['offsets'],
                                window_size):
                G.add_edge(node_i, node_j)
    # print(nx.info(G))
    return G


def produce_phrase_graphs_corpus(input_path: str, output_path: str, window_size: int):
    results = []   # store preprocessed graphs
    analysis_all = {"nodes": 0, "edges": 0, "degree": 0}
    cnt = 0
    with open(input_path) as fopen:
        for line in tqdm.tqdm(fopen):
            line_res = json.loads(line.strip())
            cnt += 1
            G = produce_phrase_graphs_per_doc(line_res, window_size)
            G.graph['docid'] = cnt     # add for further analysis
            results.append(G)
            # nx.drawing.nx_pydot.write_dot(G, "nyt.0.dot")
            nnodes = G.number_of_nodes()
            analysis_all['nodes'] += nnodes
            analysis_all['edges'] += G.number_of_edges()
            analysis_all['degree'] += (sum(dict(G.degree()).values()) / nnodes)
    for k in analysis_all:
        analysis_all[k] = analysis_all[k] / cnt
    print('total %d graphs' % (cnt))
    print(analysis_all)
    with open(output_path, 'wb') as fwrite:
        pickle.dump(results, fwrite, protocol=4)


if __name__ == '__main__':
    tree_str = "(ROOT (S (PP (IN In) (NP (DT this) (NN paper))) (NP (PRP we)) (VP (VBP present) (NP (NP (DT a) (ADJP (NP (JJ novel) (NN GRU)) (HYPH -) (VBN based)) (NN model)) (SBAR (WHNP (WDT that)) (S (VP (VBZ combines) (NP (NP (JJ syntactic) (NN information)) (ADVP (IN along) (PP (IN with) (NP (JJ temporal) (NN structure))))) (PP (IN through) (NP (DT an) (NN attention) (NN mechanism)))))))) (. .)))"
    tree = Tree.fromstring(tree_str)
    tree.pretty_print()
    print(extract_NP_VP_from_constituency_parse(tree))    # ret: list of index tuple

    # CORD Collaborate
    CORD_path = 'train.jsonlines'
    CORD_out_path = 'train.win5.pickle.gz'
    # produce_phrase_graphs_corpus(CORD_path, CORD_out_path, window_size=5)
