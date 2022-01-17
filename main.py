import torch
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
torch.manual_seed(args.seed)
np.random.seed(args.seed)

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)


# === Load model =======================
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
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()


model.load_state_dict(state_dict, strict=False)
model.to(device)

# === load data =======================
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


# === train ===========================
criterion = torch.nn.CrossEntropyLoss()
optim = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

min_loss = float('inf')
for epoch_item in range(args.start_epoch, args.epochs):
  print('=== Epoch %d ===' % epoch_item)
  train_loss = 0.
  for i, batch in enumerate(train_dataloader):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    
    optim.zero_grad()
    outputs = model.forward(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optim.step()

    train_loss += loss

    if (i+1) % args.log_interval == 0:
      with torch.no_grad():
        total_val_loss = 0.0
        model.eval()
        for j, data in enumerate(val_dataloader):
          sample, labels = data
          sample, labels = sample.to(device), labels.to(device)

          logits = model.forward(sample)
          loss = criterion(logits, labels)
          loss = loss.mean()
          total_val_loss += loss.item()

        total_val_loss /= len(val_dataloader)
        print('=== Epoch: %d/%d, Train Loss: %f, Val Loss: %f' % (
          epoch_item, i+1,  train_loss/args.log_interval, total_val_loss))
        train_loss = 0.

        # save best model
        if total_val_loss < min_loss:
          # model.save(os.path.join(args.save, "model_best.pt"))
          torch.save(model.state_dict(), os.path.join(args.save, "model_best.pt"))
          min_loss = total_val_loss
          print("Saving new best model")
  # scheduler.step()
  
# save last model
# model.save(os.path.join(args.save, "model_last.pt"))
torch.save(model.state_dict(), os.path.join(args.save, "model_last.pt"))
print("Saving new last model")










