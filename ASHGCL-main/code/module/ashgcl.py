import torch
import torch.nn as nn
import torch.nn.functional as F
from .ho_encoder import Ho_encoder
from .lo_encoder import Lo_encoder
from .contrast import Contrast
from .fs_encoder import Fs_encoder


class ASHGCL(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, fslayers, dataset):
        super(ASHGCL, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        if dataset == "acm":
            self.feat_transform = nn.Linear(feats_dim_list[0], hidden_dim)
            nn.init.xavier_normal_(self.feat_transform.weight, gain=1.414)
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.ho = Ho_encoder(P, hidden_dim, attn_drop)
        self.lo = Lo_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.fs = Fs_encoder(feats_dim_list[0], hidden_dim, hidden_dim, feat_drop, fslayers)

        self.loss_weights = nn.Parameter(torch.ones(3))
        self.global_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.weight)

        self.W_mi = nn.Parameter(torch.Tensor(feats_dim_list[0], hidden_dim))
        nn.init.xavier_uniform_(self.W_mi)
    def readout(self, z, method='mean'):
        if method == 'mean':
            return z.mean(dim=0)
        elif method == 'max':
            return z.max(dim=0)[0]

    def discriminate(self, z, summary, sigmoid=True):
        summary = torch.matmul(self.weight, summary)
        value = torch.matmul(z, summary)
        return torch.sigmoid(value) if sigmoid else value

    #Calculating global loss
    def global_loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor, readout_method='mean'):
        s = self.readout(pos_z, method=readout_method)
        h = self.global_projector(s)

        pos_loss = -torch.log(self.discriminate(pos_z, h, sigmoid=True) + 1e-8).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, h, sigmoid=True) + 1e-8).mean()
        loss = (pos_loss + neg_loss) * 0.5

        return loss

    def forward(self, feats, pos, mps, nei_index, similarity_matrix, weights,fslayers):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))

        #Encode the three views separately
        z_ho = self.ho(h_all[0], mps)
        z_lo = self.lo(h_all, nei_index)
        z_fs = self.fs(feats[0], similarity_matrix)


        #Calculating local loss
        loss_ho_lo = self.contrast(z_ho, z_lo, pos)
        loss_ho_fs = self.contrast(z_ho, z_fs, pos)
        loss_lo_fs = self.contrast(z_lo, z_fs, pos)

        #Calculate bidirectional global losses
        global_loss_ho_lo = (self.global_loss(z_ho, z_lo) + self.global_loss(z_lo, z_ho)) / 2
        global_loss_ho_fs = (self.global_loss(z_ho, z_fs) + self.global_loss(z_fs, z_ho)) / 2
        global_loss_lo_fs = (self.global_loss(z_lo, z_fs) + self.global_loss(z_fs, z_lo)) / 2

        loss = (loss_ho_lo * weights[0] +
                loss_ho_fs * weights[1] +
                loss_lo_fs * weights[2] +
                (global_loss_ho_lo * weights[3] +
                 global_loss_ho_fs * weights[4] +
                 global_loss_lo_fs * weights[5]))

        return loss

    def get_embeds(self, feats, mps):
        z_ho = F.elu(self.fc_list[0](feats[0]))
        z_ho = self.ho(z_ho, mps)
        return z_ho.detach()

