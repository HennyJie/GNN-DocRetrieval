import pandas as pd
from collections import defaultdict
import numpy as np
import os
import nltk
import json
import random

################# rnd5 doucuments #################
meta_file = "./data/SemiGIN_Dataset/2020-07-16/metadata.csv"
articles_path = './data/SemiGIN_Dataset/2020-07-16/'
valid_docids_rnds = set(line.strip() for line in open('./data/docids-rnd5.txt'))

################# training set #################
# full training set
full_training_path = './data/qrels-covid_d5_j0.5-4.txt'
doc_ids = set()

qrels = pd.read_csv(full_training_path, sep=" ", names=["qid", "jid", "docid", "label"])

################# extract text and divide into 10 folders #################
out_title_dir_path = "./data/SemiGIN_Dataset/d5_j0.5-4_title"
out_abstract_dir_path = "./data/SemiGIN_Dataset/d5_j0.5-4_abstract"
out_paragraph_dir_path = "./data/SemiGIN_Dataset/d5_j0.5-4_paragraph"

if not os.path.exists(out_title_dir_path):
    os.makedirs(out_title_dir_path)
if not os.path.exists(out_abstract_dir_path):
    os.makedirs(out_abstract_dir_path)
if not os.path.exists(out_paragraph_dir_path):
    os.makedirs(out_paragraph_dir_path)

meta_data = pd.read_csv(meta_file)

for index, row in qrels.iterrows():
    doc_id = row["docid"]

    doc_file = doc_id + '.txt'
    out_title_path = out_title_dir_path + '/' + doc_file
    out_abstract_path = out_abstract_dir_path + '/' + doc_file
    out_paragraph_path = out_paragraph_dir_path + '/' + doc_file

    if doc_id in valid_docids_rnds:
        mask = meta_data['cord_uid'] == doc_id

        # title
        if meta_data[mask]['title'] is np.nan:
            doc_title = ""
        else:
            if meta_data[mask]['title'].shape != (1,):
                doc_title = meta_data[mask]['title'][:1].item()
            else:
                doc_title = meta_data[mask]['title'].item()
        if type(doc_title) == float:
            doc_title = str("")
        title_sents = nltk.sent_tokenize(doc_title)

        if '.\n' in title_sents:
            title_sents.remove('.\n')
        sentences_out = "\n".join(title_sents)

        with open(out_title_path, 'w') as out:
            print("writing to: ", out_title_path)
            out.write(sentences_out)

        # abstract
        if meta_data[mask]['abstract'] is np.nan:
            doc_abstract = ""
        else:
            if meta_data[mask]['abstract'].shape != (1,):
                doc_title = meta_data[mask]['abstract'][:1].item()
            else:
                doc_abstract = meta_data[mask]['abstract'].item()
        if type(doc_abstract) == float:
            doc_abstract = str("")
        abstract_sents = nltk.sent_tokenize(doc_abstract)

        if '.\n' in abstract_sents:
            abstract_sents.remove('.\n')
        sentences_out = "\n".join(abstract_sents)

        with open(out_abstract_path, 'w') as out:
            print("writing to: ", out_abstract_path)
            out.write(sentences_out)

        # paragraphds
        need_check_pdf = True
        have_source = False
        paragraph_sents = []

        if meta_data[mask]['pmc_json_files'].shape != (1,):
            pmc_json_file_path = meta_data[mask]['pmc_json_files'][:1].item(
            )
        else:
            pmc_json_file_path = meta_data[mask]['pmc_json_files'].item()

        if not pd.isnull(pmc_json_file_path):
            text_file_path = articles_path + pmc_json_file_path
            if os.path.isfile(text_file_path):
                need_check_pdf = False
                have_source = True

        if need_check_pdf:
            if meta_data[mask]['pdf_json_files'].shape != (1,):
                pdf_json_file_paths = meta_data[mask]['pdf_json_files'][:1].item(
                )
            else:
                pdf_json_file_paths = meta_data[mask]['pdf_json_files'].item(
                )

            if not pd.isnull(pdf_json_file_paths):
                pdf_json_file_paths_list = pdf_json_file_paths.split(';')
                for pdf_json_file_path in pdf_json_file_paths_list:
                    text_file_path = articles_path + pdf_json_file_path
                    if os.path.isfile(text_file_path):
                        need_check_pdf = False
                        have_source = True
                        break

        if have_source:
            with open(text_file_path, 'r') as i:
                text_json = json.load(i)

            if 'body_text' in text_json:
                body_text = text_json['body_text']
                for t in body_text:
                    for sent in nltk.sent_tokenize(t['text']):
                        paragraph_sents.append(sent)

        if '.\n' in paragraph_sents:
            paragraph_sents.remove('.\n')
        sentences_out = "\n".join(paragraph_sents)

        with open(out_paragraph_path, 'w') as out:
            print("writing to: ", out_paragraph_path)
            out.write(sentences_out)
