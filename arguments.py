import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    # parser.add_argument('--cuda', dest='cuda', type=int, default=0, help='gpu id.')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=700)
    parser.add_argument('--seed', dest='seed', type=int, default=321)
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=20)

    parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=100)
    parser.add_argument('--num_gcn_layers', dest='num_gcn_layers', type=int, default=1, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--num_mlp_layers', dest='num_mlp_layers', type=int, default=1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--activation', dest='activation', type=str, default='prelu')
    parser.add_argument('--readout', dest='readout', type=str, default='mean')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01)

    parser.add_argument('--first_heads', dest='first_heads', type=int, default=2)
    parser.add_argument('--output_heads', dest='output_heads', type=int, default=1)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.6)
    parser.add_argument('--save_emb_interval', dest='save_emb_interval', type=int, default=100)
    parser.add_argument('--consistency', dest='consistency', type=int, default=1)
    return parser.parse_args()
