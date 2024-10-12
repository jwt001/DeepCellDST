from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

from .dc_model import Model as DeepCell
from .dg_model import Model as DeepGate

class TopModel(nn.Module):
    def __init__(self, 
                 args, 
                 dc_ckpt, 
                 dg_ckpt
                ):
        super(TopModel, self).__init__()
        self.args = args
        self.mask_ratio = args.mask_ratio
        
        # DeepCell 
        self.deepcell = DeepCell(dim_hidden=args.dim_hidden)
        self.deepcell.load(dc_ckpt)
        
        # DeepGate 
        self.deepgate = DeepGate(dim_hidden=args.dim_hidden)
        self.deepgate.load(dg_ckpt)
        
        # Transformer
        tf_layer = nn.TransformerEncoderLayer(d_model=args.dim_hidden * 2, nhead=args.tf_head, batch_first=True)
        self.mask_tf = nn.TransformerEncoder(tf_layer, num_layers=args.tf_layer)
        
        # Token masking
        self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden))  # learnable mask token
        
    def mask_tokens(self, tokens, mask_ratio): 
        """
        Randomly mask a ratio of tokens.
        Args:
            tokens: Input tokens (batch_size, seq_len, dim_hidden)
            mask_ratio: Percentage of tokens to mask
        Returns:
            masked_tokens: Tokens with some positions replaced by mask token
            mask_indices: Indices of masked tokens
        """
        seq_len = len(tokens)
        num_mask = int(mask_ratio * seq_len)
        mask_indices = torch.randperm(seq_len)[:num_mask]  # randomly select tokens to mask
        masked_tokens = tokens.clone()
        masked_tokens[mask_indices, self.args.dim_hidden:] = self.mask_token
        return masked_tokens, mask_indices

    def forward(self, G):
        self.device = next(self.parameters()).device
        
        # Get PM and AIG tokens
        pm_hs, pm_hf = self.deepcell(G)
        aig_hs, aig_hf = self.deepgate(G)
        pm_tokens = torch.cat([pm_hs, pm_hf], dim=1)
        aig_tokens = torch.cat([aig_hs, aig_hf], dim=1)
        mcm_pm_tokens = torch.zeros(0, self.args.dim_hidden * 2).to(self.device)
        
        # Mask a portion of PM tokens
        pm_tokens_masked, mask_indices = self.mask_tokens(pm_tokens, self.mask_ratio)
        
        # Reconstruction: Mask Circuit Modeling 
        for batch_id in range(G.batch.max().item() + 1): 
            batch_pm_tokens_masked = pm_tokens_masked[G.batch == batch_id]
            batch_aig_tokens = aig_tokens[G.aig_batch == batch_id]
            batch_all_tokens = torch.cat([batch_pm_tokens_masked, batch_aig_tokens], dim=0)
            batch_predicted_tokens = self.mask_tf(batch_all_tokens)
            batch_pred_pm_tokens = batch_predicted_tokens[:batch_pm_tokens_masked.shape[0], :]
            mcm_pm_tokens = torch.cat([mcm_pm_tokens, batch_pred_pm_tokens], dim=0)
            
        
        # Predict PM probability 
        pm_prob = self.deepcell.pred_prob(pm_hf)
        
        return mcm_pm_tokens, mask_indices, pm_tokens, pm_prob
        
    
   
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        