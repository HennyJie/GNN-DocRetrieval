import fasttext
import json 
import numpy as np 
from collections import defaultdict

model_path = './data/BioWordVec_PubMed_MIMICIII_d200.bin'
model = fasttext.load_model(model_path)

# node_dict_path = './data/train_node.json'
# node_emb_path = './data/train_node_emb.npy'
node_dict_path = './data/train_node_whole.json'
node_emb_path = './data/train_node_whole_emb.npy'

f = open(node_dict_path, "r")
all_nodes_dict = json.loads(f.read())
node_emb_dict = defaultdict(list)

for docid in all_nodes_dict:
    node_dict = all_nodes_dict[docid]
    num_nodes = len(node_dict)
    nodes_emb = []
    for i in range(num_nodes):
        word = node_dict[str(i)]
        if len(word)!=0:
            single_node_emb = model.get_word_vector(word)
        else:
            single_node_emb = np.zeros(200)
        nodes_emb.append(single_node_emb)
    print(num_nodes==len(nodes_emb))
    node_emb_dict[docid] = nodes_emb

# print(node_emb_dict)
np.save(node_emb_path, node_emb_dict, allow_pickle=True)