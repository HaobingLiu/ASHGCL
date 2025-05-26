import torch
import torch.nn as nn
import torch.nn.functional as F

#GCN aggregator within metapath
class MeanAggregator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MeanAggregator, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU()

        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        degrees = torch.spmm(adj, torch.ones(adj.size(0), 1, device=adj.device))
        out = out / (degrees + 1e-8)
        return self.act(out)

#Attention between meta-paths
class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)

            beta.append(attn_curr.matmul(sp.t()))

        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("meta-path weight:", beta.data.cpu().numpy())
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp

#High order view encoder
class Ho_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Ho_encoder, self).__init__()
        self.P = P
        self.metapath_layers = nn.ModuleList([MeanAggregator(hidden_dim, hidden_dim) for _ in range(P)])
        self.metapath_fusion = MetapathFusion(P, hidden_dim, hidden_dim)#dblp加了性能差
        self.att = Attention(hidden_dim, attn_drop)  # 创建注意力机制

    def forward(self, h, mps):
        metapath_embeds = []
        for i in range(self.P):
            metapath_embeds.append(self.metapath_layers[i](h, mps[i]))
        z_mp = self.att(metapath_embeds)  # 应用注意力机制
        return z_mp


#Intermetapath aggregation can be changed to convolution
class MetapathFusion(nn.Module):
    def __init__(self, n_metapaths, in_dim, out_dim):
        super(MetapathFusion, self).__init__()
        self.conv = nn.Parameter(torch.full((n_metapaths, in_dim), 1 / n_metapaths, dtype=torch.float32))
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h_list):
        stacked = torch.stack(h_list).transpose(0, 1)
        fused = torch.sum(stacked * self.conv, dim=1)
        return self.linear(fused), fused