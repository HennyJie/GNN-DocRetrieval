import sent2vec
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import numpy as np 
from collections import defaultdict
import json 

nltk.download('punkt')
nltk.download('stopwords')

model_path = './data/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')

stop_words = set(stopwords.words('english'))
def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)

# generate query embedding
query_path = "./data/query_dict.json"
embedding_path = "./data/query_embedding_dict.npy"

f = open(query_path, "r")
query_dict = json.loads(f.read())
embedding_dict = defaultdict(list)

for query_id in query_dict:
    query_text = query_dict[query_id]
    sentence = preprocess_sentence(query_text)
    sentence_vector = model.embed_sentence(sentence)
    embedding_dict[query_id] = sentence_vector
    print("query_id: {}, embedding: {}".format(query_id, sentence_vector))

np.save(embedding_path, embedding_dict, allow_pickle=True)