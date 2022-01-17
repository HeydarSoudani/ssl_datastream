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
  def __init__(self):
    super(MyPretrainedResnet50, self).__init__()
    
    PATH = 'moco_v2_800ep_pretrain.pth.tar'
    checkpoint = torch.load(PATH)
    state_dict = checkpoint['state_dict']
    self.pretrained = models.resnet50()

    for k in list(state_dict.keys()):
      # retain only encoder_q up to before the embedding layer
      if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
      # delete renamed or unused k
      del state_dict[k]
      
    # print(list(state_dict.keys()))

    # freeze all layers but the last fc
    for name, param in self.pretrained.named_parameters():
      print(name)
      print(name.startswith('layer4'))
      if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
    self.pretrained.load_state_dict(state_dict, strict=False)
    
    self.fc1 = nn.Linear(3072, 128)
    self.fc2 = nn.Linear(128, 10)
    
    # init the fc layers
    self.pretrained.fc.weight.data.normal_(mean=0.0, std=0.01)
    self.pretrained.fc.bias.data.zero_()
    self.fc1.apply(Xavier)
    self.fc2.apply(Xavier)
    
  
  def forward(self, x):
    # x = x.view(x.size(0), -1)
    x = self.pretrained(x)
    features = torch.relu(self.fc1(x))
    out = self.fc2(features)
    return out, features

