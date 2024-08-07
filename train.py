from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/train'

if __name__ == '__main__':
    circuit_path = os.path.join(DATA_DIR, 'graphs.npz')
    num_epochs = 60
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepcell.Model()
    
    trainer = deepcell.Trainer(model, distributed=True)
    trainer.set_training_args(loss_weight=[3.0, 1.0, 0.5], lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    