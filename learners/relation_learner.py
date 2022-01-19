import torch


class RelationLearner:
  def __init__(self, device, args):
    # self.criterion = criterion
    self.criterion = torch.nn.CrossEntropyLoss()
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }
  
  def train(self, feature_ext, batch, optimizer, iteration, args):
    feature_ext.train()
    optimizer.zero_grad()

    support_images, support_labels, query_images, query_labels = batch
    print(support_images.shape)
    print(support_labels.shape)
    print(support_labels)
    print(query_images.shape)
    print(query_labels.shape)
    print(query_labels)
  
  def evaluate(self, feature_ext, dataloader, known_labels, args):
    model.eval()