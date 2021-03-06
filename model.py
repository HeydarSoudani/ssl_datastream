import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os
import math
import subprocess

def Xavier(m):
  if m.__class__.__name__ == 'Linear':
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    if m.bias is not None:
      m.bias.data.fill_(0.0)


class MyReducedRes50(nn.Module):
  def __init__(self, args):
    super(MyReducedRes50, self).__init__()

    res = models.resnet50(pretrained=True)
    res = list(res.children())[:-3]
    self.reduced_pretrained = nn.Sequential(*res)
  
  def forward(self, x):
    # x = x.view(x.size(0), -1)
    x = self.reduced_pretrained(x)
    print(x.shape)



class MyPretrainedResnet50(nn.Module):
  def __init__(self, args):
    super(MyPretrainedResnet50, self).__init__()
    
    ## == Pretrain with SSL model MoCo V2
    # if not os.path.exists('moco_v2_800ep_pretrain.pth.tar'):
    #   subprocess.call("wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", shell=True)
    # PATH = 'moco_v2_800ep_pretrain.pth.tar'
    # checkpoint = torch.load(PATH)
    # state_dict = checkpoint['state_dict']
    # self.pretrained = models.resnet50()

    # for k in list(state_dict.keys()):
    #   # retain only encoder_q up to before the embedding layer
    #   if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #     # remove prefix
    #     state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #   # delete renamed or unused k
    #   del state_dict[k]
    # # print(list(state_dict.keys()))
    # self.pretrained.load_state_dict(state_dict, strict=False)

    ## == Pretrain with DINO
    # self.pretrained = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    # self.pretrained.fc = nn.Linear(2048, 1000)

    ## == Pretrain with torch
    self.pretrained = models.resnet50(pretrained=True)

    ## == Without Pretrain model
    # self.pretrained = models.resnet50(pretrained=False)


    ### === freeze all layers except the last fc
    for name, param in self.pretrained.named_parameters():
      # print(name)
      # print(not name.startswith(('layer4', 'fc')))
      # if name not in ['fc.weight', 'fc.bias']:
      if not name.startswith(('layer4', 'fc')):
        param.requires_grad = False
    
    self.fc1 = nn.Linear(1000, args.feature_dim)
    self.dp1 = nn.Dropout(args.dropout)
    self.fc2 = nn.Linear(args.feature_dim, args.n_classes)
    self.dp2 = nn.Dropout(args.dropout)
    
    # init the fc layers
    self.pretrained.fc.weight.data.normal_(mean=0.0, std=0.01)
    self.pretrained.fc.bias.data.zero_()
    self.fc1.apply(Xavier)
    self.fc2.apply(Xavier)
    
  def forward(self, x):
    # x = x.view(x.size(0), -1)
    x = self.pretrained(x)
    x = self.dp1(torch.relu(x))
    features = torch.relu(self.fc1(x))
    out = self.fc2(self.dp2(features))
    return out, features

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)

class MLP(nn.Module):
  def __init__(self, args, bias=True):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(2*args.feature_dim, 64)
    self.dp1 = nn.Dropout(args.dropout)
    self.fc2 = nn.Linear(64, 32)
    self.dp2 = nn.Dropout(args.dropout)
    self.fc3 = nn.Linear(32, 8)
    self.dp3 = nn.Dropout(args.dropout)
    self.fc4 = nn.Linear(8, 1)
  
  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = self.dp1(out)
    out = F.relu(self.fc2(out))
    out = self.dp2(out)
    out = F.relu(self.fc3(out))
    out = self.dp3(out)
    out = torch.sigmoid(self.fc4(out))
    return out
  
  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
    if m.bias is not None:
      m.bias.data.zero_()
  elif classname.find('BatchNorm') != -1:
    m.weight.data.fill_(1)
    m.bias.data.zero_()
  elif classname.find('Linear') != -1:
    n = m.weight.size(1)
    m.weight.data.normal_(0, 0.01)
    m.bias.data = torch.ones(m.bias.data.size())