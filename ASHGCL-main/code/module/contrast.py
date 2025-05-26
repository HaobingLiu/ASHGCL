import torch
import torch.nn as nn
import scipy.sparse as sp

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        #Temperature parameters and coefficients
        self.tau = tau
        self.lam = lam

        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    # Method of calculating similarity matrix
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_n, z_m, pos):
        z_proj_n = self.proj(z_n)
        z_proj_m = self.proj(z_m)
        matrix_n2m = self.sim(z_proj_n, z_proj_m)
        matrix_m2n = matrix_n2m.t()

        # Normalize the similarity matrix and calculate the loss
        matrix_n2m = matrix_n2m / (torch.sum(matrix_n2m, dim=1).view(-1, 1) + 1e-8)
        lori_n = -torch.log(matrix_n2m.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_m2n = matrix_m2n / (torch.sum(matrix_m2n, dim=1).view(-1, 1) + 1e-8)
        lori_m = -torch.log(matrix_m2n.mul(pos.to_dense()).sum(dim=-1)).mean()

        pos_loss = self.lam * lori_n + (1 - self.lam) * lori_m

        total_loss = pos_loss

        return total_loss
