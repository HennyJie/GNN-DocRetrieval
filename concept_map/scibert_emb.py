# generate query text
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from xml.dom.minidom import parse
import xml.dom.minidom
import json

input_path = "./data/topics-rnd5.xml"
output_path = "./data/query_dict.json"

DOMTree = xml.dom.minidom.parse(input_path)
collection = DOMTree.documentElement

topics = collection.getElementsByTagName("topic")
# out = []
query_dict = {}

for topic in topics:
    number = topic.getAttribute("number")
    query = topic.getElementsByTagName('query')[0]
    query_text = query.childNodes[0].data
    question = topic.getElementsByTagName('question')[0]
    question_text = question.childNodes[0].data
    narrative = topic.getElementsByTagName('narrative')[0]
    narrative_text = narrative.childNodes[0].data
    full_topic_text = query_text + '. ' + \
        question_text + '. ' + narrative_text + '. '

    query_dict[number] = full_topic_text

with open(output_path, 'w') as fp:
    json.dump(query_dict, fp)

# generate query embedding
query_path = "./data/query_dict.json"
embedding_path = "./data/query_embedding_dict.npy"

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
f = open(query_path, "r")
query_dict = json.loads(f.read())
embedding_dict = defaultdict(list)

for query_id in query_dict:
    query_text = query_dict[query_id]
    sentence_embeddings = model.encode(query_text)
    embedding_dict[query_id] = sentence_embeddings
    print("query_id: {}, embedding: {}".format(query_id, sentence_embeddings))

np.save(embedding_path, embedding_dict, allow_pickle=True)

# embedding_dict = np.load(embedding_path, allow_pickle=True)
# e = dict(embedding_dict.item())
# print(e)
