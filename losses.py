import torch
import torch.nn as nn
from pytorch_metric_learning import distances, losses, miners

def cos_similarity(x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception
  
  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)
  dot_prod = (x * y).sum(dim=2)
  x_norm = torch.linalg.norm(x, dim=2)
  y_norm = torch.linalg.norm(y, dim=2)
  sim = torch.div(dot_prod, x_norm*y_norm)
  
  # sim = torch.zeros((n, m))
  # for i in range(n):
  #   for j in range(m):
  #     sim[i, j] = torch.dot(x[i], y[j])/(torch.norm(x[i])*torch.norm(y[j]))

  return sim

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

    # self.metric_loss = losses.NTXentLoss(temperature=0.07)
    self.metric_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    # self.metric = losses.TripletMarginLoss(margin=0.05)
    # self.metric_loss = losses.CosFaceLoss(num_classes=args.n_classes, embedding_size=args.feature_dim, margin=0.35, scale=64)
    # self.loss_optimizer = torch.optim.SGD(self.metric_loss.parameters(), lr=0.01)
    self.ce_loss = torch.nn.CrossEntropyLoss()
    
  # def forward(self, outputs, labels, relations, labels_onehot):
  def forward(self, outputs, labels):
    metric_loss = self.metric_loss(outputs, labels.long())
    ce_loss = self.ce_loss(outputs, labels.long())
    # rel_loss = self.relation_loss(relations, labels_onehot)

    # self.loss_optimizer.step()
    # return self.lambda_1 * metric_loss
    return self.lambda_1 * metric_loss +\
           self.lambda_2 * ce_loss