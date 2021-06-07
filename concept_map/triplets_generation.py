import pandas as pd

# small traning triplets: qrels only from j3.5-4 on d5 (qrels-covid_d5_j3.5-4)
# qrels_path = "../data/qrels/qrels-covid_d5_j3.5-4.txt"
# triplets_path = "../data/triplets/triplets_r4.csv"
# no_overlap_triplets_path = "../data/triplets/no_overlap_triplets_r4.csv"

# full training triplets: qrels from j0.5-4 on d5 (qrels-covid_d5_j0.5-4)
qrels_path = "./data/qrels-covid_d5_j0.5-4.txt"
triplets_path = "./data/triplets.csv"
no_overlap_triplets_path = "./data/no_overlap_triplets.csv"

qrels = pd.read_csv(qrels_path, sep=" ", names=[
                    "qid", "jid", "docid", "label"])
qrels = qrels[["qid", "docid", "label"]]

triplets = []
no_overlap_triplets = []

positive_pairs = []
negative_pairs = []

for index, row in qrels.iterrows():
    qid = row["qid"]
    docid = row["docid"]
    label = row["label"]
    if label == 2:
        positive_pairs.append((qid, docid, label))
    elif label == 0:
        negative_pairs.append((qid, docid, label))
    else:
        pass

positive_pairs_df = pd.DataFrame(positive_pairs, columns=[
                                 "qid", "docid", "label"])
negative_pairs_df = pd.DataFrame(negative_pairs, columns=[
                                 "qid", "docid", "label"])

# all combination triplets
for positive_sample in positive_pairs:
    for negative_sample in negative_pairs:
        if positive_sample[0] == negative_sample[0]:
            triplet_sample = (
                positive_sample[0], positive_sample[1], negative_sample[1])
            triplets.append(triplet_sample)

qrels_triplets_df = pd.DataFrame(triplets, columns=[
                                 "qid", "doc+", "doc-"])
qrels_triplets_df.to_csv(triplets_path, index=False)
print("saved out triplets")

# no overlap triplets
positive_has_seen = []
negative_has_seen = []
for positive_sample in positive_pairs:
    for negative_sample in negative_pairs:
        if (positive_sample[0] == negative_sample[0]) and (positive_sample[1] not in positive_has_seen) and (negative_sample[1] not in negative_has_seen):
            triplet_sample = (
                positive_sample[0], positive_sample[1], negative_sample[1])
            no_overlap_triplets.append(triplet_sample)
            positive_has_seen.append(positive_sample[1])
            negative_has_seen.append(negative_sample[1])
            # break

qrels_triplets_small_df = pd.DataFrame(no_overlap_triplets, columns=[
    "qid", "doc+", "doc-"])
qrels_triplets_small_df.to_csv(no_overlap_triplets_path, index=False)
print("saved out no overlap triplets")
