import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

import os
import argparse
import subprocess
import numpy as np
from pandas import read_csv

from dataset import SimpleDataset
from trainer import train, test

## == Params ========================
parser = argparse.ArgumentParser()

parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'fmnist',
    'cifar10'
  ],
  default='cifar10',
  help=''
)

parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--log_interval', type=int, default=5, help='must be less then meta_iteration parameter')

# Optimizer
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=0.0005, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=5.0)
# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.5)


# Device and Randomness
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')

# Save and load model
parser.add_argument('--save', type=str, default='saved/', help='')
parser.add_argument('--best_model_path', type=str, default='saved/model_best.pt', help='')
parser.add_argument('--last_model_path', type=str, default='saved/model_last.pt', help='')

args = parser.parse_args()


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
if not os.path.exists('moco_v2_800ep_pretrain.pth.tar'):
  subprocess.call("wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", shell=True)
# subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)

PATH = 'moco_v2_800ep_pretrain.pth.tar'
checkpoint = torch.load(PATH)
state_dict = checkpoint['state_dict']
model = models.resnet50()

for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

# init the fc layer
model.fc = nn.Linear(2048, 10)
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

model.load_state_dict(state_dict, strict=False)
model.to(device)

# === Print Model layers ans params ===
print(model)

total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params: {}'.format(total_params))
print('Total trainable params: {}'.format(total_params_trainable))


## == load data =======================
print('Data loaading ...')
args.data_path = 'data/'
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)

train_data = read_csv(
  os.path.join(args.data_path, args.train_file),
  sep=',',
  header=None).values
test_data = read_csv(
  os.path.join(args.data_path, args.test_file),
  sep=',',
  header=None).values

train_dataset = SimpleDataset(train_data, args)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
test_dataset = SimpleDataset(test_data, args)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)


## == train ===========================
train(model, train_dataloader, val_dataloader, args, device)

## == Test model ======================
test(model, test_dataloader, args, device)












