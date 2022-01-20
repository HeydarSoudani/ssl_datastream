import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class W_MSE(nn.Module):
  def __init__(self):
    super().__init__()
    
  def __call__(self, output, target, weight=None):
    if weight != None:
      return torch.mean(weight * ((output-target)**2))
    else:
      return torch.mean((output-target)**2)


class W_BCE(nn.Module):
  def __init__(self):
    super().__init__()
    
  def __call__(self, output, target, weight=None):
    if weight != None:
      loss = weight*(- target * torch.log(output) - (1 - target)*torch.log(1 - output))
      return torch.mean(loss)
    else:
      loss = - target * torch.log(output) - (1 - target)*torch.log(1 - output)
      return torch.mean(loss)