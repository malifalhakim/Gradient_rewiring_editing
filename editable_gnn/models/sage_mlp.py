from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from .base import BaseGNNModel

from .gcn import GCN
from .gcn2 import GCN2
from .sage import SAGE
from .mlp import MLP

class SAGE_MLP(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 shared_weights: bool = True, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False):
        super(SAGE_MLP, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual)
        # self.alpha, self.theta = alpha, theta

        self.SAGE = SAGE(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,\
                        num_layers=num_layers, dropout=dropout, batch_norm=batch_norm, residual=residual)
        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                        batch_norm=batch_norm, residual=residual)
        
        self.mlp_freezed = True
        self.freeze_module(train=True)
        self.gnn_output = None


    def reset_parameters(self):
        ### reset GCN parameters
        for conv in self.SAGE.convs:
            conv.reset_parameters()
        if self.SAGE.batch_norm:
            for bn in self.SAGE.bns:
                bn.reset_parameters()
        
        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()
    
    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze
            
    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.SAGE, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
        else:
            self.freeze_layer(self.SAGE, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        SAGE_out = self.SAGE(x, adj_t, *args)
        MLP_out = self.MLP(x, *args)
        x = SAGE_out + MLP_out
        return x

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        # assert not self.mlp_freezed
        return self.gnn_output[idx].to(x.device) + self.MLP(x)