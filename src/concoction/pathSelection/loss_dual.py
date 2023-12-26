import torch
import torch.nn as nn
from utils import *


class AEloss(nn.Module):
    def __init__(self):
        super(AEloss, self).__init__()

    def forward(self, x, encoder, decoder, C1, C2, latent_c, \
                latent_cluster, cluster_center):

        # The Reconstructed Loss
        recon_loss = torch.sum(torch.pow((decoder - x), 2.0))

        # The L2,1 Norm of C1
        C1_loss = sparse_colmun(C1)
        diag_C1_loss = torch.sum(torch.diag(C1 ** 2.0))

        # The L2,1 Norm of C2
        C2_loss = sparse_colmun(C2)

        # Self-expression of C1
        self_C1_loss = torch.sum(torch.pow((latent_c - encoder), 2.0))

        # Self-expression of C2
        self_C2_loss = torch.sum(torch.pow((latent_cluster - cluster_center), 2.0))

        loss = {
            'recon_loss': recon_loss,
            'C1_loss': C1_loss,
            'C2_loss': C2_loss,
            'self_C1_loss': self_C1_loss,
            'diag_C1_loss': diag_C1_loss,
            'self_C2_loss': self_C2_loss
        }

        return loss
