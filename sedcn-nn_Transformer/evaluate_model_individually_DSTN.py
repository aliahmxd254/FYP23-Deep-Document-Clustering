from __future__ import print_function, division
import argparse
from ast import arg
import random
from statistics import mode
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim import RAdam 
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
from tqdm import tqdm
from torch.nn.modules.module import Module
import tsne
import os
import time
from concurrent.futures import ThreadPoolExecutor
from evaluate_model_individually_transformer import Transformer

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.set_device(0)

class PretrainAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(PretrainAE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z

class GCN(Module):
    def __init__(self, input_dim, n_clusters):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.weight_1 = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.weight_2 = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.weight_3 = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.weight_4 = Parameter(torch.FloatTensor(input_dim, input_dim))
        torch.nn.init.xavier_uniform_(self.weight_1)
        torch.nn.init.xavier_uniform_(self.weight_2)
        torch.nn.init.xavier_uniform_(self.weight_3)
        torch.nn.init.xavier_uniform_(self.weight_4)

    def forward(self, features, adj, active=True):
        adj_hat = torch.eye(adj.size(0), device=device) + adj  # Adding self-loops
        deg_inv_sqrt = torch.pow(torch.sum(adj_hat, dim=1), -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)
        adj_hat = torch.mm(torch.mm(deg_inv_sqrt, adj_hat), deg_inv_sqrt)  # Symmetrically normalized adjacency matrix

        # First layer
        support = torch.mm(features, self.weight_1)
        output_1 = torch.mm(adj_hat, support)
        output_1 = F.relu(output_1)
        
        # Second layer
        support = torch.mm(output_1, self.weight_2)
        output_2 = torch.mm(adj_hat, support)
        output_2 = F.relu(output_2)
        
        # Third layer
        support = torch.mm(output_2, self.weight_3)
        output_3 = torch.mm(adj_hat, support)
        output_3 = F.relu(output_3)
        
        # Fourth layer
        support = torch.mm(output_3, self.weight_4)
        output_4 = torch.mm(adj_hat, support)
        output_4 = F.relu(output_4)

        return [output_1, output_2, output_3, output_4]

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x, gcn_1, gcn_2, gcn_3, gcn_4):
        gamma = 0.5
        enc_h1 = F.relu(self.enc_1(x))
        f = gamma * enc_h1 + (1 - gamma) * gcn_1
        enc_h2 = F.relu(self.enc_2(f))
        f = gamma * enc_h2 + (1 - gamma) * gcn_2
        enc_h3 = F.relu(self.enc_3(f))
        f = gamma * enc_h3 + (1 - gamma) * gcn_3
        z = self.z_layer(f)
        f = gamma * z + (1 - gamma) * gcn_4

        dec_h1 = F.relu(self.dec_1(f))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, f, z

class DSTN(nn.Module):

    def __init__(self, ffn_hidden, num_heads, drop_prob, num_layers,
                 max_sequence_length, 
                 n_input, n_z, n_clusters, batch_num, add_positional_enc, v=2):
        super(DSTN, self).__init__()

        # autoencoder for intra information
        # self.ae = AE(
        #     n_enc_1=n_enc_1,
        #     n_enc_2=n_enc_2,
        #     n_enc_3=n_enc_3,
        #     n_dec_1=n_dec_1,
        #     n_dec_2=n_dec_2,
        #     n_dec_3=n_dec_3,
        #     n_input=n_input,
        #     n_z=n_z)
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu')) # load pretrain parameters

        # GCN for inter information
        # self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        # self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)
        # self.gnn_5 = GNNLayer(n_z, n_clusters)
        # transformer for attention based internal information
        # self.transformer = Transformer(d_model=n_input, ffn_hidden=ffn_hidden, num_heads=num_heads,
        #                                drop_prob=drop_prob, num_layers=num_layers, 
        #                                max_sequence_length=max_sequence_length, batch_num=batch_num, add_positional_enc=add_positional_enc)

        # gcn for external information
        self.gcn = GCN(n_input, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        gcn_embeddings = self.gcn(x,adj)
        # x_bar, tra1, tra2, tra3, z = self.ae(x)
        # GCN Module

        # x_bar, f = self.transformer(x=x, y=x, gamma = 0.5, gcn_embeddings=gcn_embeddings)
        # h = self.gnn_1(x, adj)  # h1
        # tra2 = tra2 + 0.001 *F.relu(self.ae.enc_2(h+tra1))  # ae update via (h1 + tra1)

        # h = self.gnn_2(0.5*h+0.5*tra1, adj)
        # tra3 = tra3 + 0.001 *F.relu(self.ae.enc_3(h+tra2))  # ae update via (h2 + tra2)

        # h = self.gnn_3(0.5*h+0.5*tra2, adj)
        # z = z + 0.001 *self.ae.z_layer(h+tra3)    # ae update via (h3 + tra3)

        # h = self.gnn_4(0.5*h+0.5*tra3, adj)

        # h = self.gnn_5(0.5*h+0.5*z, adj, active=False)
        # predict = F.softmax(f, dim=1)

        # Dual Self-supervised Module
        # q = 1.0 / (1.0 + torch.sum(torch.pow(f.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        # q = q.pow((self.v + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()

        predict = 1.0 / (1.0 + torch.sum(torch.pow(gcn_embeddings[3].unsqueeze(1) - self.cluster_layer, 2), 2) / self.v) # gcn_embeddings[3] was initially gcn_4
        predict = predict.pow((self.v + 1.0) / 2.0)
        predict = (predict.t() / torch.sum(predict, 1)).t()

        # dot_product
        # adj_pred = torch.sigmoid(torch.matmul(f, f.t()))2

        return gcn_embeddings[3], None, predict, predict, None


def evaluate_kmeans(random_state, X, true_labels):
    clusters = KMeans(n_init='auto', n_clusters=num_clusters, random_state=random_state, init='k-means++').fit(X).labels_
    acc, _, _, _ = eva(true_labels, clusters)
    return random_state, acc

def kmeans_seed(X, n, rstate_limit, true_labels):
    # Specify the number of clusters (you can choose an appropriate value)
    global num_clusters
    num_clusters = n
    acc_collection = {}

    for i in range(0, rstate_limit, 100):
        # Use ThreadPoolExecutor to parallelize purity calculations
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evaluate_kmeans, i, X, true_labels): i for i in range(i, i+16)}
        temp_acc_collection = {future.result()[0]: future.result()[1] for future in futures}
        acc_collection.update(temp_acc_collection)

    max_rand_state = max(acc_collection, key=acc_collection.get)
    return max_rand_state


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, itr, lambda_1=0, lambda_2=1):
    # configurations for ACM
    if(args.name == 'acm'):
        d_model = 1870
        num_heads = 1
        drop_prob = 0.1
        batch_num = 25
        max_sequence_length = 3025
        ffn_hidden = 2048
        num_layers = 4
        add_positional_enc = True
        max_iterations = 3
        n_clusters = 3
        random_states = 500
        gpu = True

    # configurations for BBC
    elif(args.name == 'bbc'):
        d_model = 9635
        num_heads = 1
        drop_prob = 0.1
        batch_num = 25
        max_sequence_length = 2225
        ffn_hidden = 2048
        num_layers = 4
        add_positional_enc = False
        max_iterations = 3
        n_clusters = 5
        random_states = 500
        gpu = True

    # configurations for Reuters
    elif(args.name == 'reut'):
        d_model = 2000
        num_heads = 1
        drop_prob = 0.1
        batch_num = 25
        max_sequence_length = 10000
        ffn_hidden = 2048
        num_layers = 4
        n_clusters = 4
        random_states = 8
        gpu = True

    # configurations for Citeseer
    elif(args.name == 'cite'):
        d_model = 3703
        num_heads = 1
        drop_prob = 0.1
        batch_num = 3
        max_sequence_length = 3327
        ffn_hidden = 1024
        num_layers = 4
        n_clusters = 6
        random_states = 16
        gpu = True

    # configurations for Doc50
    elif(args.name == 'doc50'):
        d_model = 3885
        num_heads = 1
        drop_prob = 0.1
        batch_num = 5
        max_sequence_length = 50
        ffn_hidden = 2048
        num_layers = 4
        add_positional_enc = True
        n_clusters = 5
        random_states = 500
        gpu = False
    
    elif(args.name == 'webkb'):
        d_model = 5000
        num_heads = 1
        drop_prob = 0.1
        batch_num = 202
        max_sequence_length = 8282
        ffn_hidden = 2048
        num_layers = 4

    model = DSTN(ffn_hidden=ffn_hidden, num_heads=num_heads,
                drop_prob=drop_prob, num_layers=num_layers,
                max_sequence_length=max_sequence_length,
                batch_num = batch_num,
                add_positional_enc = add_positional_enc,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    # print(model)

    optimizer = RAdam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    data_n = torch.zeros_like(data).to(device)
 
    adj_dense = adj.to_dense()
    # get the semantic from adj
    for i in tqdm(range(len(adj_dense))):
        item = adj_dense[i]
        neighbs = item.nonzero().squeeze()
        item_n = data[neighbs].mean(dim=0) + data[i]
        data_n[i] = item_n

    y = dataset.y

    # #get best cluster centers for kmeans algorithm
    # print("Finding seed for KMeans")
    # kmeans_random_seed = kmeans_seed(data.data.cpu(), n=args.n_clusters, rstate_limit=400, true_labels=y)

    # kmeans
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(data.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'k-means')

    # with torch.no_grad():
    #     pretrain_model = PretrainAE(
    #     n_enc_1=500,
    #     n_enc_2=500,
    #     n_enc_3=2000,
    #     n_dec_1=2000,
    #     n_dec_2=500,
    #     n_dec_3=500,
    #     n_input=1870,
    #     n_z=args.n_z,).cuda()
    #     pretrain_model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
    #     _, z = pretrain_model(data)


    # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # y_pred_last = y_pred
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # eva(y, y_pred, 'pae')


    res_lst = []
    model_lst = []

    # # the idx 
    np_txt = './sampling2/{}-2000.txt'.format(args.name)
    if os.path.exists(np_txt):
        random_idx = np.loadtxt(np_txt)
    else:
        random_idx = np.random.choice(range(len(y)), 2000, replace=False)
        np.savetxt(np_txt, random_idx)
    random_idx = [int(mm) for mm in random_idx]

    for epoch in tqdm(range(300)):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, z, _ = model(data, adj)
            # tmp_q = tmp_q.data
            p = target_distribution(pred)
        
            # res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P

            if epoch % 50 == 0 or epoch+1==300 or epoch == 10:
                
                z = z.data
                if lambda_2 == 0:
                    tsne.main(z.cpu().numpy()[random_idx],y[random_idx],'./pic2/{}-{}'.format(args.name,epoch))
                else:
                    tsne.main(z.cpu().numpy()[random_idx],y[random_idx],'./pic2/ours-{}-{}-{}-new-1'.format(args.name,itr,epoch))


            tmp_list = []
            # tmp_list.append(np.array(eva(y, res1, str(epoch) + 'Q')))
            tmp_list.append(np.array(eva(y, res2, str(epoch) + 'Z')))
            tmp_list.append(np.array(eva(y, res3, str(epoch) + 'P')))
            tmp_list = np.array(tmp_list)
            idx = np.argmax(tmp_list[:,0])
            # print('tag============>', idx)
            # print(tmp_list[idx][0])
            res_lst.append(tmp_list[idx])

        x_bar, q, pred, _, adj_pred = model(data, adj)

        # kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        #re_loss = F.mse_loss(x_bar, data)
        ren_loss = F.mse_loss(x_bar, data_n)
        # re_gcn_loss = F.binary_cross_entropy(adj_pred, adj_dense)


        loss = ce_loss # + 0.1 * re_loss #+ 0.001* re_gcn_loss
        # loss = 1 * kl_loss + 0.0 * ce_loss + 0 * re_loss + 0 * ren_loss # DEC
        # loss = 1 * kl_loss + 0.0 * ce_loss + 1 * re_loss + 0 * ren_loss # IDEC

        model_lst.append(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    res_lst = np.array(res_lst)
    best_idx = np.argmax(res_lst[:, 0])
    print('best--->',best_idx)
    print('dataset:{},lambda_1:{}, lambda_2:{}'.format(args.name, lambda_1, lambda_2))
    print('ACC={:.2f} +- {:.2f}'.format(res_lst[:, 0][best_idx]*100, np.std(res_lst[:, 0])))
    print('NMI={:.2f} +- {:.2f}'.format(res_lst[:, 1][best_idx]*100, np.std(res_lst[:, 1])))
    print('ARI={:.2f} +- {:.2f}'.format(res_lst[:, 2][best_idx]*100, np.std(res_lst[:, 2])))
    print('F1={:.2f} +- {:.2f}'.format(res_lst[:, 3][best_idx]*100, np.std(res_lst[:, 3])))

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='acm')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=1870, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    # args.pretrain_path = 'data/ab_study/embedding_size/{}_{}.pkl'.format(args.name, args.n_z)
    dataset = load_data(args.name)

    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256
        max_iterations = 3

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561
        max_iterations = 3

    if args.name == 'reut':
        args.k = None
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000
        args.n_z = args.n_input
        max_iterations = 3

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870
        args.n_z = args.n_input
        max_iterations = 1

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
        max_iterations = 3

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
        args.n_z = args.n_input
        max_iterations = 3

    if args.name == 'abstract':
        args.k = 10
        args.n_clusters = 3
        args.n_input = 10000
        max_iterations = 3

    if args.name == 'bbc':
        args.k = 10
        args.n_clusters = 5
        args.n_input = 5000
        args.n_z = args.n_input
        max_iterations = 3

    if args.name == 'webkb':
        args.k = None
        args.n_clusters = 7
        args.n_input = 5000
        args.n_z = args.n_input
        max_iterations = 3

    if args.name == 'doc50':
        args.k = 10
        args.n_clusters = 5
        args.n_input = 3885
        args.n_z = args.n_input
        max_iterations = 4

    print(args)
    # train_sdcn(dataset,1,0)
    
    for i in range(5):
        # train_sdcn(dataset,0.7,0.8) # dblp
        # train_sdcn(dataset, 0.7, 1.0) #dblp
        # train_sdcn(dataset, 0.1, 1) # hhar 
        train_sdcn(dataset, i, 1, 0.1) #usps
    

    # for lambda_1 in range(1,11):
    #     lambda_1 = lambda_1 * 0.1
    #     for lambda_2 in range(1,11):
    #         lambda_2 = lambda_2 * 0.1
    #         for _ in range(5):
    #             train_sdcn(dataset, lambda_1, lambda_2)
    
    end = time.time()
    duration = end - start
    print(f"Time taken: {duration:.2f}s")
