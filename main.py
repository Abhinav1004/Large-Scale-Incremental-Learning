import torch
import numpy as np
from trainer import Trainer
import sys

import os
import os.path
import torchvision
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import argparse

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
#batch size
parser.add_argument('--batch_size', default = 128, type = int) 
#epoch 
parser.add_argument('--epoch', default = 50, type = int) 
#learning rate
parser.add_argument('--lr', default = 0.1, type = int)
#max_size = total number of old data samples 
parser.add_argument('--max_size', default = 2000, type = int)
#total number of classes
parser.add_argument('--total_cls', default = 100, type = int)
args = parser.parse_args()


if __name__ == "__main__":
    trainer = Trainer(args.total_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
