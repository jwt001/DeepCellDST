from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_geometric.data import Data
from .data_utils import construct_node_feature
from .dag_utils import return_order_info
        
class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 tt_pair_index=None, tt_dis=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.y = y
        self.tt_dis = tt_dis
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        if key == 'aig_batch': 
            return 1
        if key == 'topnodes':
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'forward_index' in key or 'backward_index' in key:
            return 0
        elif 'edge_index' in key:
            return 1
        elif key == 'tt_pair_index' or key == 'connect_pair_index':
            return 1
        else:
            return 0

def parse_pyg_mlpgate(x, edge_index, y, num_gate_types=3):
    x_torch = construct_node_feature(x, num_gate_types)#对于每个节点的门的种类，生成one hot编码  torch.tensor([[0, 1, 0], [1, 0, 0]])

    # tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    # tt_pair_index = tt_pair_index.t().contiguous()
    # rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    # rc_pair_index = rc_pair_index.t().contiguous()
    # tt_dis = torch.tensor(tt_dis)
    # is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    if len(edge_index) == 0:
        edge_index = edge_index.t().contiguous()
        forward_index = torch.LongTensor([i for i in range(len(x))])
        backward_index = torch.LongTensor([i for i in range(len(x))])
        forward_level = torch.zeros(len(x))
        backward_level = torch.zeros(len(x))
    else:
        edge_index = edge_index.t().contiguous()
        print("edge_index =", edge_index.shape)
        forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))

    graph = OrderedData(x=x_torch, edge_index=edge_index, y = y,
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index)
    graph.use_edge_attr = False

    # add reconvegence info
    # graph.rec = torch.tensor(x[:, 3:4], dtype=torch.float)
    # graph.rec_src = torch.tensor(x[:, 4:5], dtype=torch.float)
    # add gt info
    # add indices for gate types
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)#每个节点对应一个门
    graph.prob = torch.tensor(y).reshape((len(x), 1))
    graph.topnodes = torch.tensor(len(x) - 1, dtype=torch.long)
    return graph
