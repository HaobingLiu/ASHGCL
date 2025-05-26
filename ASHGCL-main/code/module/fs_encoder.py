
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj):
        support = self.linear(x)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return output

class ImprovedGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, n_layers=2):
        super(ImprovedGCN, self).__init__()
        self.layers = nn.ModuleList([GCNLayer(nfeat if i == 0 else nhid, nhid) for i in range(n_layers-1)])
        self.output_layer = GCNLayer(nhid, nclass)
        self.dropout = dropout
        self.residual_linear = nn.ModuleList([nn.Linear(nfeat if i == 0 else nhid, nhid) for i in range(n_layers-1)])

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            #residual = x
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            #if x.shape[1] != residual.shape[1]:
                #residual = self.residual_linear[i](residual)
            #x += residual  # 添加残差连接
        x = self.output_layer(x, adj)
        return x


#Feature similarity view encoder
class Fs_encoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, n_layers=2):
        super(Fs_encoder, self).__init__()
        self.gcn = ImprovedGCN(nfeat, nhid, nclass, dropout, n_layers)

    def forward(self, x, adj):
        return self.gcn(x, adj)