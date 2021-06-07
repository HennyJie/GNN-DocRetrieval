import numpy as np
import logging
import os.path as osp
import os
import random
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, PReLU
import torch_geometric.transforms as T
import pandas as pd
from collections import defaultdict
import collections
from collections import defaultdict
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import paired_distances
# from arguments import arg_parse
from utils import get_activation, seed_everything
from net import MLP, GCN, Net
from simplified_net import SimplifiedMLP, SimplifiedGCN, NodePooling
from dataloader import get_train_loader, get_test_loader
from loss import TripletMarginWithDistanceLoss
import subprocess
import nni 
from sp import SimpleParam
import argparse

data_folder = '../DATA_Standardized'
query_emb_path = 'query_embedding_dict.npy'


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    epoch_loss = 0

    for i, data in enumerate(train_loader):
        # each data should include three elements in a triplet (qid, doc+ and doc-)
        qid = data[0]
        query_emb = data[1].squeeze()
        positive_docid = data[2]
        positive_graph = data[3]
        negative_docid = data[4]
        negative_graph = data[5]

        query_emb = query_emb.to(device)
        positive_graph = positive_graph.to(device)
        negative_graph = negative_graph.to(device)

        optimizer.zero_grad()

        # triplet loss
        positive_emb = model(positive_graph.x, positive_graph.batch, device)
        negative_emb = model(negative_graph.x, negative_graph.batch, device)

        triplet_loss = TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=1.0, swap=True, reduction='mean')
        loss = triplet_loss(query_emb, positive_emb, negative_emb)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    return epoch_loss

@torch.no_grad()
def test(epoch, final=False):
    model.eval()

    ranking_list = []
    fake_rank_value = 0

    for data in test_loader:
        qid = data[0].item()
        query_emb = data[1].reshape(1, -1)

        docid = data[2][0]
        doc_graph = data[3]
        doc_graph = doc_graph.to(device)
        doc_emb = model(doc_graph.x, doc_graph.batch, device)
        
        doc_emb_data = doc_emb.detach().cpu().numpy()
        score = F.cosine_similarity(torch.Tensor(query_emb), torch.Tensor(doc_emb_data)).item()

        ranking_list.append([qid, "Q0", docid, fake_rank_value, score, "run"])
    
    ranking_df = pd.DataFrame(ranking_list, columns=["qid", "Q0", "docid", "rank", "score", "tag"])
    ranking_df["docid"] = ranking_df["docid"].map(lambda x: x.split('.')[0])
    ranking_df["rank"] = ranking_df.groupby("qid")["score"].rank("first", ascending=False)
    ranking_df = ranking_df.groupby(["qid"]).apply(lambda x: x.sort_values(["rank"], ascending=True))

    ranking_out_dirs = Path(data_folder) / Path('eval/node_pooling')
    if not os.path.exists(ranking_out_dirs):
        os.makedirs(ranking_out_dirs)
    ranking_out_name = f"node_pooling_epoch{epoch}_batch{batch_size}_gc{num_gcn_layers}_mlp{num_mlp_layers}_lr{learning_rate}_{activation}_{readout}.txt"
    out_path = Path(ranking_out_dirs) / Path(ranking_out_name)

    ranking_df.to_csv(out_path, sep=' ', index=False, header=False)

    # evaluation
    test_qrels_path = "../DATA_Standardized/eval/qrels-covid_d5_j4.5-5.txt"
    eval_excutable = "../trec_eval-9.0.7/trec_eval"
    if os.path.exists(eval_excutable):  
        rc, out = subprocess.getstatusoutput(eval_excutable + " -m " + " ndcg_cut.20 " + test_qrels_path + " " + str(out_path))
        ndcg_20 = float(out.split('\t')[-1])
        print(f'node pooling ndcg_20: {ndcg_20}')

    if final and use_nni:
        nni.report_final_result(ndcg_20)
    elif use_nni:
        nni.report_intermediate_result(ndcg_20)
    
    return ndcg_20


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--param', type=str, default='local:node.yaml')
    # parser.add_argument('--cuda', dest='cuda', type=int, default=0, help='gpu id.')
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=20)
    parser.add_argument('--save_emb_interval', dest='save_emb_interval', type=int, default=100)
    parser.add_argument('--seed', dest='seed', type=int, default=54321)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=700)
    default_param = {
            'batch_size': 32,
            'epoch_num': 20,
            'activation': "relu",
            'num_GCN_layers': 2,
            'num_MLP_layers': 1,
            'readout': "mean",
            'learning_rate': 0.01,
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    use_nni = args.param == 'nni'
    
    eval_interval = args.eval_interval
    save_emb_interval = args.save_emb_interval
    seed_everything(args.seed)
    hidden_dim = args.hidden_dim

    batch_size = param['batch_size']
    epoch_num = param['epoch_num']

    num_gcn_layers = param['num_GCN_layers']
    num_mlp_layers = param['num_MLP_layers']
    learning_rate = param['learning_rate']
    activation = param['activation']
    readout = param['readout']

    # query emb
    query_embedding = dict(np.load(Path(data_folder) / Path(query_emb_path), allow_pickle=True).item())
    for k, v in query_embedding.items():
        query_embedding[k] = torch.from_numpy(v)

    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    node_feature_dim = 200
    dataset_num_features = node_feature_dim
    model =  NodePooling(
        SimplifiedGCN(dataset_num_features, hidden_dim, num_gcn_layers, get_activation(activation), readout),
        SimplifiedMLP(num_gcn_layers * hidden_dim, hidden_dim, num_mlp_layers, get_activation(activation)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_train_loader(data_folder, query_embedding, batch_size)
    test_loader = get_test_loader(data_folder, query_embedding)
    
    best_ndcg_20 = 0
    best_params = {}
    for epoch in range(1, epoch_num+1):
        train_loss = train(epoch)
        print("Epoch {} loss: {}".format(epoch, train_loss))

        if epoch % eval_interval == 0:
            ndcg_20 = test(epoch)
            if best_ndcg_20 < ndcg_20:
                best_ndcg_20 = ndcg_20
                param['epoch_num'] = epoch
                best_params = param

    ndcg_20 = test(epoch, final=True)
    if ndcg_20 > best_ndcg_20:
        best_ndcg_20 = ndcg_20
        param['epoch_num'] = epoch
        best_params = param
    
    print(f'N pool final_ndcg_20: {ndcg_20}')
    print(f'N pool best_ndcg_20: {best_ndcg_20}')
    print(f'N pool best_params: {best_params}')
