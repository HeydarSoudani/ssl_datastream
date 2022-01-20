import torch
import time
from losses import W_MSE, W_BCE

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
    self.criterion = W_MSE()
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }
  
  def train(self, feature_ext, relation_net, batch, optimizer, iteration, args):
    feature_ext.train()
    relation_net.train()
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
    support_features = features[:support_len]
    query_features = features[support_len:] #[w, 128]

    support_features_ext = support_features.unsqueeze(0).repeat(args.ways, 1, 1)  #[w, w*sh, 128]
    support_features_ext = torch.transpose(support_features_ext, 0, 1)            #[w*sh, w, 128]
    support_labels = support_labels.unsqueeze(0).repeat(args.ways, 1)             #[w, w*sh]
    support_labels = torch.transpose(support_labels, 0, 1)                        #[w*sh, w]

    query_features_ext = query_features.unsqueeze(0).repeat(args.ways*args.shot, 1, 1) #[w*sh, w, 128]
    query_labels = query_labels.unsqueeze(0).repeat(args.ways*args.shot, 1)            #[w*sh, w]

    relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, args.feature_dim*2) #[w*w*sh, 256]
    relarion_labels = torch.zeros(args.ways*args.shot, args.ways).to(device)
    relarion_labels = torch.where(
      support_labels!=query_labels,
      relarion_labels,
      torch.tensor(1.).to(device)
    ).view(-1,1)

    print(relation_pairs.shape)
    print(relarion_labels)
    time.sleep(5)

    ### === Relation Network ===========================
    # relations = relation_net(relation_pairs)

    # loss = criterion(relations, relarion_labels, weight=relarion_weights)
    
    # model_optim.zero_grad()
    # mclassifer_optim.zero_grad()
    # loss.backward()
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
    # torch.nn.utils.clip_grad_norm_(mclassifer.parameters(),0.5)
    # model_optim.step()
    # mclassifer_optim.step()

    ### === Loss & backward ============================
    loss = self.criterion(outputs[:support_len], support_labels) # without Relation network


    loss.backward()
    torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
    optimizer.step()

    return loss.detach().item()


  def evaluate(self, feature_ext, dataloader, known_labels, args):
    feature_ext.eval()

    known_labels = torch.tensor(list(known_labels), device=self.device)
    pts = torch.cat(
      [self.prototypes[l.item()] for l in known_labels]
    )
    
    with torch.no_grad():
      total_loss = 0.0
      total_dist_acc = 0.0
      correct_cls_acc = 0.0
      total_cls_acc = 0

      for i, batch in enumerate(dataloader):
        samples, labels = batch
        labels = labels.flatten()
        samples, labels = samples.to(self.device), labels.to(self.device)
        logits, features = feature_ext.forward(samples)

        ## == Distance-based Acc. ============== 
        dists = torch.cdist(features, pts)  #[]
        argmin_dists = torch.min(dists, dim=1).indices
        pred_labels = known_labels[argmin_dists]
        
        acc = (labels==pred_labels).sum().item() / labels.size(0)
        total_dist_acc += acc

        ## == Cls-based Acc. ===================
        _, predicted = torch.max(logits, 1)
        total_cls_acc += labels.size(0)
        correct_cls_acc += (predicted == labels).sum().item()

        ## == loss =============================
        # unique_label = torch.unique(labels)
        # prototypes = torch.cat(
        #   [self.prototypes[l.item()] for l in unique_label]
        # )
        # loss = self.criterion(
        #   features,
        #   logits,
        #   labels,
        #   prototypes,
        #   n_query=args.query_num,
        #   n_classes=args.ways,
        # )
        # total_loss += loss.item()

        loss = self.criterion(logits, labels)
        loss = loss.mean()
        total_loss += loss.item()

      total_loss /= len(dataloader)
      total_dist_acc /= len(dataloader)
      total_cls_acc = correct_cls_acc / total_cls_acc  

      return total_loss, total_dist_acc, total_cls_acc