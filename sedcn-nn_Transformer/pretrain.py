import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from evaluation2 import eva
import random 
import os
import sys

sys.path.insert(0, r"D:\FAST\FYP\FYP23-Deep-Document-Clustering\sedcn-nn_Transformer")


from utils_transformer.Encoder import Encoder
from utils_transformer.PositionalEncoding import PositionalEncoding

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


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(300):
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

        torch.save(model.state_dict(), 'bbc.pkl')

# model = AE(
#         n_enc_1=500,
#         n_enc_2=500,
#         n_enc_3=2000,
#         n_dec_1=2000,
#         n_dec_2=500,
#         n_dec_3=500,
#         n_input=9635,
#         n_z=10,).cuda()
        

dataset_name = "bbc"

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
