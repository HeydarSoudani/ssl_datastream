import torch
import torch.nn as nn
import torchvision.models as models
import math

def Xavier(m):
  if m.__class__.__name__ == 'Linear':
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    if m.bias is not None:
      m.bias.data.fill_(0.0)

class MyPretrainedResnet50(nn.Module):
  def __init__(self, args):
    super(MyPretrainedResnet50, self).__init__()
    
    # Pretrain with SSL model MoCo V2
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
    # print(list(state_dict.keys()))

    # Pretrain with torch
    self.pretrained = models.resnet50(pretrained=True)

    # freeze all layers but the last fc
    for name, param in self.pretrained.named_parameters():
      print(name)
      print(not name.startswith(('layer4', 'fc')))
      # if name not in ['fc.weight', 'fc.bias']:
      if not name.startswith(('layer4', 'fc')):
        param.requires_grad = False
    self.pretrained.load_state_dict(state_dict, strict=False)
    
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

