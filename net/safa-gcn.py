import torch
import torch.nn as nn

from .sa_gcn import Model as SA_GCN
from .fa_gcn import Model as FA_GCN


class T_Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, embeddings):
        e = self.attention(embeddings)
        return e

class Fusion(nn.Module):
    """
    Score-level and Clip-level Fusion
    """
    def __init__(self, in_channels, out_channels, graph_args, args1=(), kwargs1={}, args2=(), kwargs2={}, dropout_context=0.2, dropout_gcn=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sag = SAG(in_channels, out_channels, beta_dim=beta_dim, graph_args=graph_args, dropout=dropout_gcn)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=out_channels, out_features=2, bias=True),
        )

        self.temporal_attn = T_Attention(in_channels=out_channels, hid_dim=64)

        self.spatial_stream = SA_GCN(*args1, **kwargs1)
        self.frequency_stream = FA_GCN(*args2, **kwargs2)


    def forward(self, x, f):
        n_blocks = x.size(1)
        n_channels = x.size(2)
        block_size = x.size(3)
        n_joints = x.size(4)
        x = x.view(x.size(0) * n_blocks, n_channels, block_size, n_joints)
        # stream-fusion
        f = torch.fft.rfft(x, dim=2) 
        x_emb = torch.cat((self.spatial_stream(x), self.frequency_stream(f)), 1)
        
        # clip-fusion
        x_emb = x_emb.view(x.size(0), n_blocks, self.out_channels).contiguous()
        e = self.temporal_attn(x_emb)

        lambda_ = torch.softmax(e, dim=1)

        context = torch.bmm(torch.transpose(lambda_, 1, 2), x_emb).squeeze(1)
        # without applying non-linearity
        scores = self.output(context)

        return scores

