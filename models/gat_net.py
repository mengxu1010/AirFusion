import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from models.gat_layer import GATLayer
from models.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params[0] # node_dim (feat is an integer)
        hidden_dim = net_params[1]
        out_dim = net_params[2]
        n_classes = net_params[2]
        num_heads = 8
        dropout = 0.6
        n_layers = 1

        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        
        self.layers = nn.ModuleList([GATLayer(in_dim_node, hidden_dim, num_heads,
                                              dropout, self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, 0, self.graph_norm, self.batch_norm, self.residual))

    def forward(self, g, h, snorm_n, snorm_e):

        # GAT
        for conv in self.layers:
            h = conv(g, h, snorm_n)
            
        return h
    

class GATNet_ss(nn.Module):

    def __init__(self, net_params, num_par, num_heads, dropouts, n_layers, timestep, categories):
        super().__init__()
        d_in = net_params[0]-2
        d_in += len(categories) * net_params[5]
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories), net_params[5])
        nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        r = self.category_embeddings.weight
        in_dim_node = d_in * timestep
        hidden_dim1 = net_params[1]
        hidden_dim2 = net_params[2]
        hidden_dim3 = net_params[3]
        out_dim = net_params[4]
        num_heads1 = num_heads[0]
        num_heads2 = num_heads[1]
        num_heads3 = num_heads[2]
        dropout1 = dropouts[0]
        dropout2 = dropouts[1]
        dropout3 = dropouts[2]
        layers1 = n_layers[0]
        layers2 = n_layers[1]
        layers3 = n_layers[2]

        self.graph_norm = False
        self.batch_norm = False
        self.residual = False
        self.LSTM = nn.LSTM(in_dim_node, hidden_dim1, num_layers=layers1, batch_first=True, bidirectional=True)
        self.layers1 = nn.ModuleList([GATLayer(hidden_dim1*2, hidden_dim2, num_heads1,
                                              dropout1, self.graph_norm, self.batch_norm, self.residual) for _ in range(layers2)])
        self.layers2 = nn.ModuleList([GATLayer(hidden_dim2*num_heads1, hidden_dim3, num_heads2,
                                              dropout3, self.graph_norm, self.batch_norm, self.residual) for _ in range(layers3)])
        self.layers2.append(GATLayer(hidden_dim3 * num_heads2, out_dim, num_heads3, dropout2, self.graph_norm, self.batch_norm, self.residual))
        self.classifier_ss = nn.Linear(hidden_dim3 * num_heads2, num_par, bias=False)

    def forward(self, g_od, g_sem, _h_num, _h_cat, snorm_n, snorm_e):
        _h = []
        if _h_num is not None:
            _h.append(_h_num)
        if _h_cat is not None:
            _h_cat_e = self.category_embeddings(_h_cat + self.category_offsets[None]).view(_h_cat.size(0), _h_cat.size(1), _h_cat.size(2), -1)

            _h_cat_e.requires_grad_()
            _h.append(_h_cat_e)
        _h = torch.cat(_h, dim=-1)

        _h = _h.permute(0, 2, 1, 3)
        print (_h.shape)
        _h = _h.reshape(_h.shape[0], _h.shape[1], _h.shape[2]*_h.shape[3])
        h, _ = self.LSTM(_h)
        # GAT
        for conv in self.layers1:
            h = conv(g_od, h, snorm_n)
            h = h.permute(1, 0, 2)
        for conv in self.layers2:
            h_ss = h
            h = conv(g_sem, h, snorm_n)
            h = h.permute(1, 0, 2)
        h_ss = self.classifier_ss(h_ss)

        return h, h_ss, _h_cat_e, self.category_embeddings.weight
 
