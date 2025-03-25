from .gcn import GCN
from .sage import SAGE
from .mlp import MLP

from .gcn_mlp import GCN_MLP
from .sage_mlp import SAGE_MLP

__all__ = [
    'GCN',
    'SAGE',
    'GRAPHSAINT',
    'MLP',
    'GCN_MLP',
    'SAGE_MLP',
    'GRAPHSAINT_MLP'
]