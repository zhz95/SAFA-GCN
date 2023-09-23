import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net.utils.sgraph import Sgraph

class SAGCN(nn.Module):
    r"""Spatial-Attention Graph Convolutional Networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Sgraph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        kernel_size = A.size(0)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.s_gcn_networks = nn.ModuleList((
            s_gcn(in_channels, 32, 1, residual=False, **kwargs0),
            s_gcn(32, 64, 2, **kwargs),

        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.s_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.s_gcn_networks)


        self.spatial_attn = Attention(in_dim=64, hid_dim=64)

        # fcn for prediction	

        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.s_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.view(N, c, t, v).permute(0, 1, 2, 3)#
        x = F.avg_pool2d(x, (x.size()[2], 1))
        x = x.squeeze(dim=2).permute(0, 2, 1)
	
        e = self.spatial_attn(x)
        beta = torch.softmax(e, dim=1)
        x = torch.bmm(torch.transpose(beta, 1, 2), x).permute(0, 2, 1)
        x = x.unsqueeze(dim=3)


        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)


        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.s_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.relu(self.linear2(x))


class ConvSpatialGraphical(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1), 
            bias=bias
        )

    def forward(self, x, A):

        x = self.conv(x)

        n, kc, t, v = x.size()     
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A

class Attention(nn.Module):
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

class s_gcn(nn.Module):
    r"""Applies a spatial-attention graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the graph convolving kernel
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual=True):
        super().__init__()


        self.gcn = ConvSpatialGraphical(in_channels, out_channels)

        self.mlp = MLP(in_channels, out_channels)


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)

        x, A = self.gcn(x, A)

        N, C, T, V = x.size()
        x = x.permute(0, 2, 3, 1).reshape(N*T*V, C)
        x = self.mlp(x)
        x = x.reshape(N, T, V, -1).permute(0, 3, 1, 2)
        


        return self.relu(x + res), A
