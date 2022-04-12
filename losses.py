import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners


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


class TotalLoss(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.lambda_1 = args.lambda_1
    self.lambda_2 = args.lambda_2
    
    self.relation_loss = torch.nn.MSELoss()
    self.ce_loss = torch.nn.CrossEntropyLoss()
    self.metric_loss = losses.NTXentLoss(temperature=0.07)
    # self.metric = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    # self.metric = losses.TripletMarginLoss(margin=0.05)

  def forward(self, outputs, labels, relations, labels_onehot):
  # def forward(self, outputs, labels):
    metric_loss = self.metric_loss(outputs, labels.long())
    ce_loss = self.ce_loss(outputs, labels.long())
    # rel_loss = self.relation_loss(relations, labels_onehot)

    # return self.lambda_1 * metric_loss
    return self.lambda_1 * metric_loss +\
           self.lambda_2 * ce_loss