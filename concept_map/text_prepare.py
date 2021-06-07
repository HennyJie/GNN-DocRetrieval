import pandas as pd
from collections import defaultdict
import numpy as np
import os
import nltk
import json
import random

################# rnd5 doucuments #################
meta_file = "./data/SemiGIN_Dataset/metadata.csv"
articles_path = './data/SemiGIN_Dataset/2020-07-16/'
valid_docids_rnds = set(line.strip() for line in open('./data/docids-rnd5.txt'))
triplet_path = './data/no_overlap_triplets.csv'

################# training set #################
doc_ids = set()
triplet = pd.read_csv(triplet_path, sep=",", names=["qid", "doc+", "doc-"])

for index, row in triplet.iterrows():
    doc_ids.add(row["doc+"])
    doc_ids.add(row["doc-"])

################# extract text and divide into 10 folders #################
out_dir_path = "./data/SemiGIN_Dataset/trainset"

meta_data = pd.read_csv(meta_file)

for index, doc_id in enumerate(doc_ids):
    print(doc_id)
    doc_file = doc_id + '.txt'
    out_path = out_dir_path + '/' + doc_file

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
        title_sents = doc_title

        # abstract
        if meta_data[mask]['abstract'] is np.nan:
            doc_abstract = ""
        else:
            if meta_data[mask]['abstract'].shape != (1,):
                doc_abstract = meta_data[mask]['abstract'][:1].item()
            else:
                doc_abstract = meta_data[mask]['abstract'].item()
        if type(doc_abstract) == float:
            doc_abstract = str("")
        abstract_sents = doc_abstract

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
                    if len(t) != 0:
                        paragraph_sents.append(t['text'])

        with open(out_path, 'w') as out:
            out.write(title_sents)
            if (len(title_sents) != 0) and (not (title_sents[-1] == "\n")):
                out.write("\n")

            out.write(abstract_sents)
            if (len(abstract_sents) != 0) and (not (abstract_sents[-1] == "\n")):
                out.write("\n")

            for sent in paragraph_sents:
                out.write(sent)
                if (len(sent) != 0) and not (sent[-1] == "\n"):
                    out.write("\n")
