"""GraphSAGE / GAT encoder + bilinear link decoder.

    h_v^(k) = sigma( W^(k) · CONCAT( h_v^(k-1), AGGREGATE_{u in N(v)} h_u^(k-1) ) )       (GraphSAGE, mean)
    P(u, v) = sigmoid( z_u^T W_p z_v )                                                    (decoder)

W_p is symmetrised, (W_p + W_p^T)/2, because co-occurrence links are undirected.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, num_layers: int = 2,
                 model_type: str = "sage", heads: int = 4, dropout: float = 0.3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        model_type = model_type.lower()
        if model_type not in ("sage", "gat"):
            raise ValueError("model_type must be 'sage' or 'gat'")
        if model_type == "gat" and hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GAT")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            last = i == num_layers - 1
            if model_type == "sage":
                self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr="mean"))
            else:
                if last:
                    self.convs.append(GATConv(dims[i], dims[i + 1], heads=1, dropout=dropout))
                else:
                    self.convs.append(GATConv(dims[i], dims[i + 1] // heads, heads=heads, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x) if isinstance(conv, GATConv) else F.relu(x)
        return x


class BilinearDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)

    def sym_w(self):
        return 0.5 * (self.W + self.W.t())

    def forward(self, z, pairs):
        """pairs: LongTensor (k, 2) -> logits (k,)"""
        zu, zv = z[pairs[:, 0]], z[pairs[:, 1]]
        return ((zu @ self.sym_w()) * zv).sum(dim=-1)

    def score_one_to_many(self, z, idx: int):
        """logits between node `idx` and every node -> (N,)"""
        return (z[idx] @ self.sym_w()) @ z.t()


class LinkPredictionModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64, num_layers: int = 2,
                 model_type: str = "sage", heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers,
                           model_type=model_type, heads=heads, dropout=dropout)
        self.encoder = GNNEncoder(**self.config)
        self.decoder = BilinearDecoder(out_dim)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, pairs):
        return self.decoder(z, pairs)

    def forward(self, x, edge_index, pairs):
        return self.decode(self.encode(x, edge_index), pairs)

    @classmethod
    def from_config(cls, config: dict) -> "LinkPredictionModel":
        return cls(**config)
