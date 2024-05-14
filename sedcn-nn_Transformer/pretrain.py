import os
import sys
import numpy as np
import h5py
import torch
import random 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from utils_transformer.Encoder import Encoder
from utils_transformer.PositionalEncoding import PositionalEncoding

# sys.path.insert(0, r"D:\FAST\FYP\FYP23-Deep-Document-Clustering\sedcn-nn_Transformer")
sys.path.insert(0, r"..")


seed = 42

os.environ['PYTHONHASHSEED']=str(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#torch.cuda.set_device(3)


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

class LinearLayer(nn.Module):
    def __init__(self, n_input, n_z):
        super().__init__()
        self.z_layer = Linear(n_input, n_z).cuda()
    
    def forward(self, x):
        x_bar = self.z_layer(x)
        return x_bar

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)

            loss = F.mse_loss(x_bar.squeeze(), x.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, attention_list = model(x) #attention_list = z

            linear_layer = LinearLayer(d_model, n_z=n_z)
            z = linear_layer(attention_list[-1].squeeze())
            
            loss = F.mse_loss(x_bar.squeeze(), x.squeeze())
            print('{} loss: {}'.format(epoch, loss))           
            kmeans = KMeans(n_clusters=5, n_init=10, random_state=seed).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

        # save_dir = '..\data/'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        torch.save(model.state_dict(), f'data/{dataset_name}.pkl')

# model = AE(
#         n_enc_1=500,
#         n_enc_2=500,
#         n_enc_3=2000,
#         n_dec_1=2000,
#         n_dec_2=500,
#         n_dec_3=500,
#         n_input=9635,
#         n_z=10,).cuda()
        

dataset_name = "webkb"

n_z = 10

if dataset_name == "bbc":
    max_sequence_length = 2225 #dataset.x.shape[0]
    d_model = 9635 #dataset.x.shape[1]
    ffn_hidden = 1024
    num_heads = 1
    drop_prob = 0.1
    num_layers = 3
    num_batches = 25

elif dataset_name == "doc50":
    max_sequence_length = 50 #dataset.x.shape[0]
    d_model = 3885 #dataset.x.shape[1]
    ffn_hidden = 2048
    num_heads = 1
    drop_prob = 0.1
    num_layers = 5
    num_batches = 25
    
elif dataset_name == 'webkb':
    d_model = 5000
    num_heads = 1
    drop_prob = 0.1
    num_batches = 202
    max_sequence_length = 8282
    ffn_hidden = 2048
    num_layers = 3


model = Encoder(d_model=d_model, ffn_hidden=ffn_hidden, num_heads=num_heads, drop_prob=drop_prob, num_layers=num_layers).cuda()

x = np.loadtxt(f'D:\FAST\FYP\FYP23-Deep-Document-Clustering\data\{dataset_name}.txt', dtype=float)
y = np.loadtxt(f'D:\FAST\FYP\FYP23-Deep-Document-Clustering\data\{dataset_name}_label.txt', dtype=int)

pe = PositionalEncoding(d_model=d_model, max_sequence_length=max_sequence_length)
positional_encoding = pe.forward()

dim_diff = d_model - positional_encoding.shape[1]  

if dim_diff != 0:
    positional_encoding = positional_encoding[:,0:dim_diff]

x = torch.from_numpy(x) + positional_encoding
x = x.numpy()

print(x.shape)


try:
    x = x.reshape((1, x.shape[0], x.shape[1]))
except Exception as e:
    raise Exception(e)

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)
