import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms

import os
import argparse
import numpy as np
from pandas import read_csv

from model import MyPretrainedResnet50
from dataset import SimpleDataset
from phases.batch_learn import batch_learn
from phases.init_learn import init_learn

from visualize import visualization

## == Params ========================
parser = argparse.ArgumentParser()

parser.add_argument(
  '--phase',
  type=str,
  choices=[
    'batch_learn',
    'init_learn',
    'zeroshot_test',
    'stream_learn',
    'zeroshot_test_base',
    'batch_incremental_learn',
    'episodic_incremental_learn',
    'plot'
  ],
  default='plot',
  help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'fmnist',
    'cifar10',
    'cifar100',
  ],
  default='cifar10',
  help=''
)

parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--log_interval', type=int, default=5, help='must be less then meta_iteration parameter')
parser.add_argument('--meta_iteration', type=int, default=3000, help='')

# Sampler
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=5, help='')
parser.add_argument('--query_num', type=int, default=5, help='')

# Optimizer
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=0.0005, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=5.0)
# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.5)

# Model
parser.add_argument('--feature_dim', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.5)

# Device and Randomness
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')

# Save and load model
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--best_model_path', type=str, default='saved/model_best.pt', help='')
parser.add_argument('--last_model_path', type=str, default='saved/model_last.pt', help='')

args = parser.parse_args()

## == Set class number =================
if args.dataset in ['mnist', 'fmnist', 'cifar10']:
  args.n_classes = 10
elif args.dataset in ['cifar100']:
  args.n_classes = 100

## == Device ===========================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Device: {}'.format(device))

## == Apply seed =======================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Load model =======================
print('Pretrain model loading ...')
model = MyPretrainedResnet50(args)
model.to(device)

# === Print Model layers ans params ====
# print(model)
total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params: {}'.format(total_params))
print('Total trainable params: {}'.format(total_params_trainable))

## == load data ========================
print('Data loaading ...')
args.data_path = 'data/'
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)

## == training =========================
if __name__ == '__main__':
  ## == Batch learning ===
  if args.phase == 'batch_learn':
    batch_learn(model, args, device)
  ## == Data Stream ======
  elif args.phase == 'init_learn':
    init_learn(model, args, device)

## == visualization ===================
# visualization(model, test_dataset, args, device)











