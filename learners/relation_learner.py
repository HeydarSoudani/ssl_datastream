import torch


class RelationLearner:
  def __init__(self, device, args):
    # self.criterion = criterion
    self.criterion = torch.nn.CrossEntropyLoss()
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(args.n_classes)
    }
  
  def train(self, feature_ext, batch, optimizer, iteration, args):
    feature_ext.train()
    optimizer.zero_grad()

    print(batch)

  
  def evaluate(self, feature_ext, dataloader, known_labels, args):
    model.eval()