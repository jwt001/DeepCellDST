from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
import deepcell.top_model
import deepcell.top_trainer 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/dg_pair'

if __name__ == '__main__':
    args = get_parse_args()
    circuit_path = '/Users/zhengyuanshi/studio/DeepCell_Dataset/deepgate_dataset/pair_graphs.npz'
    num_epochs = 60
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepcell.top_model.TopModel(
        args, 
        dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt='./ckpt/dg.pth'
    )
    
    trainer = deepcell.top_trainer.TopTrainer(args, model, distributed=True)
    trainer.set_training_args(lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    