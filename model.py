import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader as TorchDataLoader

import torch_geometric.nn as gnn
from torch_scatter import scatter_sum

from transformers import AutoModel


class GNNL(gnn.MessagePassing):
    def __init__(self, d_model, num_heads, drop=0.1, arch='sgnn', use_pred=True):
        super(GNNL, self).__init__(aggr='add')
        self.d_model = d_model
        self.num_heads = num_heads
        self.arch = arch
        self.use_pred = use_pred

        if use_pred:
            self.ck = nn.Linear(2 * d_model, d_model)
        else:
            self.qk = nn.Linear(d_model, d_model)

        self.qw = nn.Linear(d_model, d_model)
        self.vw = nn.Linear(d_model, d_model)

        self.aw = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(num_heads, num_heads)
        )

        self.drop = nn.Dropout(drop)

        self.cw = nn.Linear(d_model, d_model)

        if arch == 'sgnn':
            self.lx = nn.Linear(d_model, d_model)

            self.la = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LeakyReLU(0.2),
                nn.Linear(d_model, 2),
                nn.Softmax(dim=1),
                nn.Dropout(drop)
            )

    def forward(self, x, edge_index, edge_attr):
        q = self.qw(x[edge_index[1]])

        if self.use_pred:
            k = self.ck(torch.cat([edge_attr, x[edge_index[0]]], dim=1))
        else:
            k = self.qk(x[edge_index[0]])

        attention = self._attention(q, k, edge_index, x.shape[0]).unsqueeze(-1)

        h = self.propagate(edge_index, x=x, edge_attr=edge_attr, attention=attention)

        if self.arch == 'gnn':
            return h + x

        lx = self.lx(x)

        la = self.la(torch.cat([lx, h], dim=1))

        return x * la[:, 0].unsqueeze(-1) + h * la[:, 1].unsqueeze(-1)

    def _attention(self, q, k, ei, n):
        wq = self._reshape_mh(q)
        wk = self._reshape_mh(k)

        aw = torch.einsum('bhd,bhq->bh', wq, wk) / math.sqrt(self.d_model)
        a = torch.exp(self.aw(aw))

        sc = scatter_sum(a, ei[1], dim=0, dim_size=n)

        ad = sc[ei[1]]
        ad[ad == 0] = 1

        return self.drop(a / ad)

    def message(self, x_i, x_j, edge_attr, attention):
        v = self.vw(x_j)

        wv = self._reshape_mh(v)

        fw = attention * wv

        return self.cw(self._reshape_out(fw))

    def _reshape_mh(self, x):
        return x.view(x.shape[0], self.num_heads, -1)

    def _reshape_out(self, x):
        return x.reshape(x.shape[0], self.d_model)


class GNN(nn.Module):
    def __init__(self, d_model, num_heads, drop=0.1, d=3, arch='sgnn', use_pred=True):
        super(GNN, self).__init__()
        self.gnns = nn.ModuleList([GNNL(d_model, num_heads, drop, arch=arch, use_pred=use_pred) for _ in range(d)])

    def forward(self, x, edge_index, edge_attr):
        for g in self.gnns:
            x = g(x, edge_index, edge_attr)
        return x


class BertEmb(nn.Module):
    def __init__(self, model):
        super(BertEmb, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.max_seq_len = 512
        self.bert.pooler.requires_grad_(False)

    def forward(self, x):
        mask = x > 0

        if x.size(1) > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            mask = mask[:, :self.max_seq_len]

        out = self.bert(input_ids=x, attention_mask=mask)['last_hidden_state']

        om = mask.unsqueeze(-1).float()

        mo = out * om
        cf = om.sum(dim=1)
        cf[cf == 0] = 1
        return mo.sum(dim=1) / cf


class Model(nn.Module):
    def __init__(self, model, d=1, arch='sgnn', use_pred=True):
        super(Model, self).__init__()
        self.arch = arch
        self.use_pred = use_pred

        self.emb1 = BertEmb(model)
        self.gnn = GNN(768, 8, d=d, arch=arch, use_pred=use_pred)

    def forward(self, cqa=None, positive_sbg=None, negative_sbg=None):

        if cqa is not None:
            cqa = self.embed_cqa(cqa)

        sbg = None

        if positive_sbg is not None:
            x, xi, edge_index, edge_attr, edge_attr_i = positive_sbg

            sbg = self.embed_subg(x, xi, edge_index, edge_attr, edge_attr_i)
        nsbg = None

        if negative_sbg is not None:
            nx, nxi, nedge_index, nedge_attr, nedge_attr_i = negative_sbg
            nsbg = self.embed_subg(nx, nxi, nedge_index, nedge_attr, nedge_attr_i)

        return cqa, sbg, nsbg

    def embed_cqa(self, x):
        return self.emb1(x)

    def embed_subg(self, x, xi, edge_index, edge_attr, edge_attr_i):
        feats = []
        for f in TorchDataLoader(x, batch_size=4, shuffle=False):
            feats.append(checkpoint.checkpoint(self.emb1, f, use_reentrant=False))
        feats = torch.cat(feats, dim=0)
        sf = feats[xi]
        if self.arch == 'lm':
            return sf

        edge_attr_sf = None
        if self.use_pred:
            props = []
            for f in TorchDataLoader(edge_attr, batch_size=4, shuffle=False):
                props.append(checkpoint.checkpoint(self.emb1, f, use_reentrant=False))
            props = torch.cat(props, dim=0)
            edge_attr_sf = props[edge_attr_i]

        out = checkpoint.checkpoint(self.gnn, sf, edge_index, edge_attr_sf, use_reentrant=False)
        return out
