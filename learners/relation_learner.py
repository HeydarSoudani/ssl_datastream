import torch

def compute_prototypes(
  support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
  """
  Compute class prototypes from support features and labels
  Args:
    support_features: for each instance in the support set, its feature vector
    support_labels: for each instance in the support set, its label
  Returns:
    for each label of the support set, the average feature vector of instances with this label
  """
  seen_labels = torch.unique(support_labels)

  # Prototype i is the mean of all instances of features corresponding to labels == i
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )

class RelationLearner:
  def __init__(self, device, args):
    # self.criterion = criterion
    self.criterion = torch.nn.CrossEntropyLoss()
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }
  
  def train(self, feature_ext, relation, batch, optimizer, iteration, args):
    feature_ext.train()
    optimizer.zero_grad()
    
    ### === Prepare data ===============================
    support_len = args.ways * args.shot 
    support_images, support_labels, query_images, query_labels = batch
    
    unique_label = torch.unique(support_labels)

    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    images = torch.cat((support_images, query_images))
    labels = torch.cat((support_labels, query_labels))

    ### === Feature extractor ===========================
    outputs, features = feature_ext.forward(images)

    ### === Prototypes ==================================
    episode_prototypes = compute_prototypes(
      features[:support_len], support_labels
    )
    old_prototypes = torch.cat(
      [self.prototypes[l.item()] for l in unique_label]
    )
    beta = args.beta * iteration / args.meta_iteration
    new_prototypes = beta * old_prototypes + (1 - beta) * episode_prototypes

    ### === Concat features ============================
    support_feature = features[:support_len]
    query_feature = features[support_len:]


    ### === Relation Network ===========================

    ### === Loss & backward ============================
    loss = self.criterion(outputs, labels)   # without Relation network


    loss.backward()
    torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
    optimizer.step()


  def evaluate(self, feature_ext, dataloader, known_labels, args):
    feature_ext.eval()