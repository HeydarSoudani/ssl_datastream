import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception
  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


class DCELoss(nn.Module):
  def __init__(self, device, gamma=0.05):
    super().__init__()
    self.gamma = gamma
    self.device = device

  def forward(self, features, labels, prototypes, n_query, n_classes):
    unique_labels = torch.unique(labels)
    features = torch.cat(
      [features[(labels == l).nonzero(as_tuple=True)[0]] for l in unique_labels]
    )

    dists = euclidean_dist(features, prototypes)
    # dists = (-self.gamma * dists).exp() 

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = (
      torch.arange(0, n_classes, device=self.device, dtype=torch.long)
      .view(n_classes, 1, 1)
      .expand(n_classes, n_query, 1)
    )

    loss_val = -log_p_y.gather(2, target_inds).mean()
    return loss_val


class TotalLoss(nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1
    self.lambda_2 = args.lambda_2
    
    self.dce = DCELoss(device, gamma=args.temp_scale)
    self.ce = torch.nn.CrossEntropyLoss()

  def forward(self, features, outputs, labels, prototypes, n_query, n_classes):
    dce_loss = self.dce(features, labels, prototypes, n_query, n_classes)
    cls_loss = self.ce(outputs, labels.long())

    return self.lambda_1 * dce_loss +\
           self.lambda_2 * cls_loss