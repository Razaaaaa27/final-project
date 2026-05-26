# model_def.py
# Definisi arsitektur GIN dan GINE untuk klasifikasi inhibitor BACE-1.
#
# Mendukung dua arsitektur:
#   - GIN  : Graph Isomorphism Network standar (tanpa fitur edge)
#   - GINE : GIN with Edge features (Hu et al., 2020)
#
# Kedua model memakai JK-Concat readout dan classifier MLP yang sama
# untuk perbandingan adil.
#
# Catatan: arsitektur di file ini harus konsisten persis dengan yang
# digunakan saat training. Jangan modifikasi.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, BatchNorm


# ============================================================
# GIN — tanpa fitur edge
# ============================================================

class GINModel(nn.Module):
    # Graph Isomorphism Network untuk klasifikasi graf molekuler.
    # Args:
    #   node_feat_dim: Dimensi fitur input per node (43 untuk skema OGB)
    #   hidden_dim:    Dimensi hidden layer
    #   num_layers:    Jumlah lapisan GIN conv
    #   num_classes:   Jumlah kelas output (2 untuk binary classification)
    #   dropout:       Dropout rate

    def __init__(self, node_feat_dim, hidden_dim=256, num_layers=4,
                 num_classes=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))

        # JK-Concat: input dim = hidden_dim * (num_layers + 1)
        concat_dim = hidden_dim * (num_layers + 1)

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.node_encoder(x)
        h_list = [global_add_pool(h, batch)]

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_list.append(global_add_pool(h, batch))

        h_graph = torch.cat(h_list, dim=1)
        return self.classifier(h_graph)


# ============================================================
# GINE — dengan fitur edge
# ============================================================

class GINEModel(nn.Module):
    # GIN with Edge Features (Hu et al., 2020).
    # Edge features ditambahkan ke node features di dalam pesan agregasi.
    # Args:
    #   node_feat_dim: Dimensi fitur node (43)
    #   edge_feat_dim: Dimensi fitur edge (12)
    #   hidden_dim:    Dimensi hidden layer
    #   num_layers:    Jumlah lapisan GINE conv
    #   num_classes:   Jumlah kelas output (2)
    #   dropout:       Dropout rate

    def __init__(self, node_feat_dim, edge_feat_dim,
                 hidden_dim=256, num_layers=4,
                 num_classes=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        # Edge encoder per layer (edge_feat_dim -> hidden_dim)
        self.edge_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.edge_encoders.append(
                nn.Linear(edge_feat_dim, hidden_dim)
            )
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))

        concat_dim = hidden_dim * (num_layers + 1)

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch = data.batch

        h = self.node_encoder(x)
        h_list = [global_add_pool(h, batch)]

        for i in range(self.num_layers):
            e = self.edge_encoders[i](edge_attr)
            h = self.convs[i](h, edge_index, e)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h_list.append(global_add_pool(h, batch))

        h_graph = torch.cat(h_list, dim=1)
        return self.classifier(h_graph)


# ============================================================
# Factory function
# ============================================================

def create_model(model_type, config):
    # Buat instance model berdasarkan tipe dan config.
    # Args:
    #   model_type: 'GIN' atau 'GINE'
    #   config:     dict berisi parameter arsitektur
    # Returns: instance model
    if model_type == 'GIN':
        return GINModel(
            node_feat_dim=config['node_feat_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config.get('num_classes', 2),
            dropout=config['dropout'],
        )
    elif model_type == 'GINE':
        return GINEModel(
            node_feat_dim=config['node_feat_dim'],
            edge_feat_dim=config['edge_feat_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config.get('num_classes', 2),
            dropout=config['dropout'],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Pilih 'GIN' atau 'GINE'.")