import torch

def compute_prototypes(
  support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:

  seen_labels = torch.unique(support_labels)

  # Prototype i is the mean of all instances of features corresponding to labels == i
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )

class PtLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }

  def train(self, model, batch, optimizer, args):
    model.train()  
    optimizer.zero_grad()

    ### === Prepare data ===============================
    support_len = args.shot * args.ways
    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[2:])
    query_labels = query_labels.flatten()
    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    images = torch.cat((support_images, query_images))
    
    ### === Feature extractor ==========================
    outputs, features = model.forward(images)
    
    episode_prototypes = compute_prototypes(
      features[:support_len], support_labels
    )
    
    ### === Loss & backward ============================
    loss = self.criterion(
      features[support_len:],
      outputs[support_len:],
      query_labels,
      episode_prototypes,
      n_query=args.query_num,
      n_classes=args.ways,
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    return loss.detach().item()

  def evaluate(
    self,
    model,
    val_loader,
    known_labels,
    args):
    model.eval()
    
    ### === Params =====================================
    total_loss = 0.0
    cw_total = 0.0
    cw_correct = 0.0
    ow_total = 0.0
    ow_correct = 0.0 
    
    ce = torch.nn.CrossEntropyLoss()

    known_labels = torch.tensor(list(known_labels), device=self.device)
    pts = torch.cat(
      [self.prototypes[l.item()] for l in known_labels]
    )

    with torch.no_grad():
      for i, batch in enumerate(val_loader):
        samples, labels = batch
        labels = labels.flatten()
        samples, labels = samples.to(self.device), labels.to(self.device)
        
        ### === Feature extractor ==============
        outputs, features = model.forward(samples)

        ## == Distance-based Acc. ============== 
        dists = torch.cdist(features, pts)  #[]
        argmin_dists = torch.min(dists, dim=1).indices
        predicted_labels = known_labels[argmin_dists]
        
        ow_total += labels.size(0)
        ow_correct += (predicted_labels == labels).sum().item()

        ## == Close World Acc. =================
        _, cw_predict_labels = torch.max(outputs, 1)
        cw_total += labels.size(0)
        cw_correct += (cw_predict_labels == labels).sum().item()

        ## == CE loss ==========================
        loss = ce(outputs, labels)
        loss = loss.mean()
        total_loss += loss.item()

      total_loss /= len(val_loader)
      total_cw_acc = cw_correct / cw_total
      total_ow_acc = ow_correct / ow_total 

      return total_loss, total_cw_acc, total_ow_acc

  def calculate_prototypes(self, model, dataloader):
    model.eval()
    
    all_features = []
    all_labels = []
    with torch.no_grad():
      for j, data in enumerate(dataloader):
        sample, labels = data
        sample, labels = sample.to(self.device), labels.to(self.device)
        _, features = model.forward(sample)
        all_features.append(features)
        all_labels.append(labels)
      
      all_features = torch.cat(all_features, dim=0)
      all_labels = torch.cat(all_labels, dim=0)
      
      unique_labels = torch.unique(all_labels)
      pts = compute_prototypes(all_features, all_labels)
      for idx, l in enumerate(unique_labels):
        self.prototypes[l.item()] = pts[idx].reshape(1, -1).detach()

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)






