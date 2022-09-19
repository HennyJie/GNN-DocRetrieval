# GNN-DocRetrieval

## Dataset and Repo Organization

This is the code base for paper: How Can Graph Neural Networks Help Document Retrieval: A Case Study on CORD19 with Concept Map Generation.

The official datasets we used in our study and the detailed information of each file can be found in the link below.
- CORD: <https://github.com/allenai/cord19>

This code repo is organized as below, 
```
|-- concept_map/
    |-- constituency_preprocess.java
    |-- constituency_parser.py
    |-- ...
    |-- ...
|-- net.py
|-- main_gat.py
|-- main_E_pool.py
|-- ...
```
where the files in the subfolder `\concept_map` are about the concept map construction and utility analyses, while files in the outer directory are models and our pipelines. 


## Usage Guideline

### Stage 1: Data Preparation
After downloading the raw text data from CORD, you need to run the following preprocessings for the preparation of concept map generation. 
- *Generate triplets for the training set*: we generate the training triplets formated as `(query, positive_doc, negative_doc)` from the last-round officially released relevance label `qrels-covid_d5_j0.5-4.txt`. The script to use is `triplets_generation.py` under the `\concept_map` subfolder. 

- *BM25 preselect for the testing set*: the indexing we implemented is based on BM25 algorithm from the Standard [Lucene](https://lucene.apache.org/core/) Library. The index fields include all full text: title + abstract + paragraphs. The script `bm25_index_preprocess.py` is used to collect text for indexing. All documents from the final round should be included in indexing and those appeared in relevence judgements are removed in order to avoid information leakage. 

### Stage 2: Concept Map Construction
- *Parsing and coreference clustering (Time Consuming)*: for each document, we perform tokenize, lemma, constituency parse and coreference clustering based on the [StandfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) repo. See script `constituency_preprocess.java` for detail. Note that all raw texts should be processed to ensure there is no latex code noise embedded. Otherwise it would raise a recurrent error in syntax tree parsing.  
  
- *Use constituency parser and other essential tools to generate phrase graph*: we further perform extract_NP_VP_from_constituency_parse, is_within_window, produce_phrase_graphs_per_doc, produce_phrase_graphs_corpus on the results of the previous step for concept map construction. See `constituency_parser.py` for a small demo example. 

- *Format graph/node for GNN*: process the concept map for the standard input format of the graph models. The node id is unified for the neural network process and the utility analyses.

- *Node feature initialization and query embedding generation*: we initiate the node embedding and generate the query embedding based on a pretrained model [BioWordVec & BoiSentVec](https://github.com/ncbi-nlp/BioSentVec). See scripts `biosent2vec.py`, `bioword2vec.py` and `scibert_emb.py` (tested) for detail. 
  
### Stage 3: Models
The code supports passing parameters through command line, using an external `param.json` file, or training with nni. You can run different `main_{model}.py` files for different models. For more information about the parameters in the pipeline, please refer to the usage below:

```bash
usage: supervised_model_gat.py [-h] [--param PARAM] [--eval_interval EVAL_INTERVAL] 
                              [--save_emb_interval SAVE_EMB_INTERVAL] [--seed SEED] 
                              [--hidden_dim HIDDEN_DIM] [--batch_size [BATCH_SIZE]]
                              [--epoch_num [EPOCH_NUM]] [--activation [ACTIVATION]] 
                              [--num_GCN_layers [NUM_GCN_LAYERS]] [--num_MLP_layers [NUM_MLP_LAYERS]] 
                              [--readout [READOUT]] [--learning_rate [LEARNING_RATE]] 
                              [--first_heads [FIRST_HEADS]] [--output_heads [OUTPUT_HEADS]] 
                              [--dropout [DROPOUT]]

optional arguments:
  -h, --help            show this help message and exit
  --param PARAM
  --eval_interval EVAL_INTERVAL
  --save_emb_interval SAVE_EMB_INTERVAL
  --seed SEED
  --hidden_dim HIDDEN_DIM
  --batch_size [BATCH_SIZE]
  --epoch_num [EPOCH_NUM]
  --activation [ACTIVATION]
  --num_GCN_layers [NUM_GCN_LAYERS]
  --num_MLP_layers [NUM_MLP_LAYERS]
  --readout [READOUT]
  --learning_rate [LEARNING_RATE]
  --first_heads [FIRST_HEADS]
  --output_heads [OUTPUT_HEADS]
  --dropout [DROPOUT]
```

### Stage 4: Evaluate
For the evaluation part, we adopt [trec_eval](https://github.com/usnistgov/trec_eval), the standard tool used by the TREC community for
evaluating an ad hoc retrieval run, given the results file and a standard set of judged results. The specific metrics we used in this paper (e.g. ndcg_cut.20, etc.) are based on the competition requirements. Note that since the evaluation has been integrated into the pipeline for model evaluation, please make sure that trec_eval has been installed locally before running the models. For separate testing, you can use:

`$ ./trec_eval [-q] [-m measure] qrel_file results_file`. 

See the official documents for more usage. 


## Requirements

- python 3.8.5
- pytorch 1.6.0
- torchvision 0.7.0
- torch-geometric 1.6.3
- torch-cluster 1.5.8
- torch-scatter 2.0.5
- torch-sparse 0.6.8
- networkx 2.5
- numpy 1.19.2
- pandas 1.2.0
- scikit-learn 0.24.0
- nltk 3.4.4
- pickleshare 0.7.5
- json-tricks 3.15.5
- graphviz 2.42.3
- pyyaml 5.3.1
- scipy 1.5.2
- cython 0.29.21
- tqdm 4.55.0
- nni 2.0


# Citation

Please cite our paper if you find this code useful for your work:

```
@inproceedings{cui2022can,
  title={How Can Graph Neural Networks Help Document Retrieval: A Case Study on CORD19 with Concept Map Generation},
  author={Cui, Hejie and Lu, Jiaying and Ge, Yao and Yang, Carl},
  booktitle={European Conference on Information Retrieval},
  pages={75--83},
  year={2022},
  organization={Springer}
}
```

