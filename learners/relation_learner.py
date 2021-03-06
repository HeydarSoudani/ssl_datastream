import torch
import random
import numpy as np
from losses.metric_loss import cos_similarity

def compute_prototypes(
  support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
  seen_labels = torch.unique(support_labels)
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )

class RelationLearner:
  def __init__(self, extractor_criterion, relation_criterion, device, args):
    self.extractor_criterion = extractor_criterion
    self.relation_criterion = relation_criterion
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }
    self.examplers = {
      l: torch.zeros(args.n_examplers, args.feature_dim, device=device)
      for l in range(args.n_classes)
    }
  
  def end2end_3d_train(
    self,
    feature_ext,
    relation_net,
    batch,
    feature_ext_optim,
    relation_net_optim,
    miteration_item,
    known_labels,
    args
  ):
    feature_ext.train()
    relation_net.train()
    feature_ext_optim.zero_grad()
    relation_net_optim.zero_grad()

    n_known = len(known_labels)
    rep_per_class = 1

    ### === Prepare data ===============================
    support_len = args.ways * args.shot 
    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[1:])
    query_labels = query_labels.flatten()

    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    unique_labels = torch.unique(support_labels).to(self.device)
    # print(support_images.shape)
    # print(query_images.shape)
    images = torch.cat((support_images, query_images))
    labels = torch.cat((support_labels, query_labels))
    
    ### === Feature extractor ==========================
    outputs, features = feature_ext.forward(images)
    support_features = features[:support_len] #[w*s, 128]
    query_features = features[support_len:]   #[w*q, 128]

  def end2end_train(
    self,
    feature_ext,
    relation_net,
    batch,
    feature_ext_optim,
    relation_net_optim,
    iteration,
    known_labels,
    args
  ):
    feature_ext.train()
    relation_net.train()
    feature_ext_optim.zero_grad()
    relation_net_optim.zero_grad()

    n_known = len(known_labels)
    rep_per_class = 1

    ### === Prepare data ===============================
    support_len = args.ways * args.shot 
    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[1:])
    query_labels = query_labels.flatten()

    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    unique_labels = torch.unique(support_labels).to(self.device)
    # print(support_images.shape)
    # print(query_images.shape)
    images = torch.cat((support_images, query_images))
    labels = torch.cat((support_labels, query_labels))
    
    ### === Feature extractor ==========================
    outputs, features = feature_ext.forward(images)
    support_features = features[:support_len] #[w*s, 128]
    query_features = features[support_len:]   #[w*q, 128]

    
    if args.rep_approach == 'prototype':
      episode_pts = compute_prototypes(support_features, support_labels)
      for idx, l in enumerate(unique_labels):
        self.prototypes[l.item()] = episode_pts[idx].reshape(1, -1).detach()
  
      reps = torch.cat([self.prototypes[l.item()] for l in known_labels])

    elif args.rep_approach == 'exampler':
      for idx, l in enumerate(unique_labels):
        k = random.sample(list(np.arange(args.shot)), 1)
        self.examplers[l.item()] = support_features[int(idx*args.shot)+k[0]].reshape(1, -1).detach()
      
      reps = torch.cat([self.examplers[l.item()] for l in known_labels])
      
    known_labels = torch.tensor(list(known_labels), device=self.device)

    ### === Concat features ============================
    support_features_ext = reps.unsqueeze(0).repeat(args.query_num, 1, 1)     #[q, w*sh, 128]
    support_features_ext = torch.transpose(support_features_ext, 0, 1)                    #[w*sh, q, 128]
    support_labels_ext = known_labels.unsqueeze(0).repeat(args.query_num, 1)    #[q, w*sh]
    support_labels_ext = torch.transpose(support_labels_ext, 0, 1)                        #[w*sh, q]

    query_features_ext = query_features.unsqueeze(0).repeat(n_known*rep_per_class, 1, 1)        #[w*sh, q, 128]
    query_labels_ext = query_labels.unsqueeze(0).repeat(n_known*rep_per_class, 1)               #[w*sh, ]
    
    relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]
    relarion_labels = torch.zeros(n_known*rep_per_class, args.query_num).to(self.device)
    relarion_labels = torch.where(
      support_labels_ext!=query_labels_ext,
      relarion_labels,
      torch.tensor(1.).to(self.device)
    ).view(-1,1)

    ### === Relation Network ===========================
    relations = relation_net(relation_pairs).view(-1, n_known) #[w, w*q]
    
    ### === Loss & backward ============================
    quety_label_pressed = torch.tensor([(known_labels == l).nonzero(as_tuple=True)[0] for l in query_labels], device=self.device)
    query_labels_onehot = torch.zeros(
      args.query_num, n_known
    ).to(self.device).scatter_(1, quety_label_pressed.view(-1,1), 1)
    query_labels_onehot = query_labels_onehot.to(self.device)

    loss = self.relation_criterion(
      relations,
      query_labels_onehot
    )
    loss.backward()

    torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_norm_(relation_net.parameters(), args.grad_clip)
    feature_ext_optim.step()
    relation_net_optim.step()

    return loss.detach().item()

  def feature_ext_train(
    self,
    feature_ext,
    batch,
    feature_ext_optim,
    args
  ):
    feature_ext.train()
    feature_ext_optim.zero_grad()

    ### === Prepare data ===============================
    support_images, support_labels, _, _ = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()

    support_images = support_images.to(self.device)
    support_labels = support_labels.to(self.device)

    ### === Feature extractor ==========================
    support_outputs, _ = feature_ext.forward(support_images)

    ### === Loss & backward ============================
    loss = self.extractor_criterion(
      support_outputs,
      support_labels
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
    feature_ext_optim.step()
    
    return loss.detach().item()

  def relation_train(
    self,
    feature_ext,
    relation_net,
    batch,
    relation_net_optim,
    known_labels,
    args
  ):
    feature_ext.eval()
    relation_net.train()
    relation_net_optim.zero_grad()

    ### === Prepare representors ======================
    n_known = len(known_labels)
    if args.rep_approach == 'prototype':
      rep_per_class = 1
      support_features = torch.cat(
        [self.prototypes[l.item()] for l in known_labels]
      )
    elif args.rep_approach == 'exampler':
      rep_per_class = args.n_examplers
      support_features = torch.cat(
        [self.examplers[l.item()] for l in known_labels]
      )
    known_labels = torch.tensor(list(known_labels), device=self.device)
    # support_labels = known_labels
    support_labels = torch.transpose(known_labels.repeat(rep_per_class).reshape(rep_per_class, -1), 0, 1).flatten()

    ### === Prepare data ===============================
    _, _, query_images, query_labels = batch
    query_images = query_images.reshape(-1, *query_images.shape[1:])
    query_labels = query_labels.flatten()
    query_images = query_images.to(self.device)
    query_labels = query_labels.to(self.device)

    ### === Feature extractor ==========================
    with torch.no_grad():
      _, query_features = feature_ext.forward(query_images)

    ### === Concat features ============================
    support_features_ext = support_features.unsqueeze(0).repeat(args.query_num, 1, 1)     #[q, w*sh, 128]
    support_features_ext = torch.transpose(support_features_ext, 0, 1)                    #[w*sh, q, 128]
    support_labels_ext = support_labels.unsqueeze(0).repeat(args.query_num, 1)    #[q, w*sh]
    support_labels_ext = torch.transpose(support_labels_ext, 0, 1)                        #[w*sh, q]

    query_features_ext = query_features.unsqueeze(0).repeat(n_known*rep_per_class, 1, 1)        #[w*sh, q, 128]
    query_labels_ext = query_labels.unsqueeze(0).repeat(n_known*rep_per_class, 1)               #[w*sh, ]

    relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]
    relarion_labels = torch.zeros(n_known*rep_per_class, args.query_num).to(self.device)
    relarion_labels = torch.where(
      support_labels_ext!=query_labels_ext,
      relarion_labels,
      torch.tensor(1.).to(self.device)
    ).view(-1,1)

    ### === Relation Network ===========================
    relations = relation_net(relation_pairs).view(-1,n_known*rep_per_class) #[w, w*q]
    relations_mean = torch.mean(
      torch.stack(
        torch.split(relations, n_known, dim=1),
        dim=2
      ), 2
    )

    ### === Loss & backward ============================
    quety_label_pressed = torch.tensor(
      [(known_labels == l).nonzero(as_tuple=True)[0] for l in query_labels],
      device=self.device
    )
    query_labels_onehot = torch.zeros(
      args.query_num, n_known
    ).to(self.device).scatter_(1, quety_label_pressed.view(-1,1), 1)
    query_labels_onehot = query_labels_onehot.to(self.device)
    loss = self.relation_criterion(
      relations_mean,
      query_labels_onehot
    )
    loss.backward()

    torch.nn.utils.clip_grad_norm_(relation_net.parameters(), args.grad_clip)
    relation_net_optim.step()

    return loss.detach().item()

  def calculate_prototypes(self, feature_ext, dataloader):
    feature_ext.eval()
    
    all_features = []
    all_labels = []
    with torch.no_grad():
      for j, data in enumerate(dataloader):
        samples, labels = data
        samples, labels = samples.to(self.device), labels.to(self.device)
        _, features = feature_ext.forward(samples)
        all_features.append(features)
        all_labels.append(labels)
      
      all_features = torch.cat(all_features, dim=0)
      all_labels = torch.cat(all_labels, dim=0)
      
      unique_labels = torch.unique(all_labels)
      pts = compute_prototypes(all_features, all_labels)
      for idx, l in enumerate(unique_labels):
        self.prototypes[l.item()] = pts[idx].reshape(1, -1).detach()

  def set_items_per_label(self, dataset):
    self.items_per_label = {}
    for item, label in enumerate(dataset.labels):
      if label in self.items_per_label.keys():
        self.items_per_label[label].append(item)
      else:
        self.items_per_label[label] = [item]

  def calculate_examplers(self, feature_ext, dataset, k=1):
    feature_ext.eval()

    with torch.no_grad():
      for label in self.items_per_label.keys():
        self.examplers[label] = torch.cat(
          [
            feature_ext(dataset[idx][0].unsqueeze(0).to(self.device))[1]
            for idx in random.sample(self.items_per_label[label], k)
          ]
        )

  def evaluate(
    self,
    feature_ext,
    relation_net,
    val_loader,
    known_labels,
    args
  ):
    feature_ext.eval()
    relation_net.eval()

    ### === Params =====================================
    total_loss = 0.0
    cw_total = 0.0
    cw_correct = 0.0
    ow_total = 0.0
    ow_correct = 0.0 

    ### === Prepare PTs ================================
    n_known = len(known_labels)
    known_labels = torch.tensor(list(known_labels), device=self.device)
    
    if args.rep_approach == 'prototype':
      rep_per_class = 1
      sup_features = torch.cat(
        [self.prototypes[l.item()] for l in known_labels]
      )
    elif args.rep_approach == 'exampler':
      rep_per_class = args.n_examplers
      sup_features = torch.cat(
        [self.examplers[l.item()] for l in known_labels]
      )
    sup_labels = torch.transpose(known_labels.repeat(rep_per_class).reshape(rep_per_class, -1), 0, 1).flatten()
    

    with torch.no_grad():
      for i, batch in enumerate(val_loader):
        
        # Query set
        test_images, test_labels = batch
        test_labels = test_labels.flatten()
        test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)

        ### === Feature extractor ==============
        test_outputs, test_features = feature_ext.forward(test_images)

        ## == Calculate accuracy ===============
        if args.similarity_approach == 'relation':
          ## == Relation Network preparation =====
          sup_features_ext = sup_features.unsqueeze(0).repeat(args.query_num, 1, 1)  #[q, nc*sh, 128]
          sup_features_ext = torch.transpose(sup_features_ext, 0, 1)                 #[nc*sh, q, 128]
          sup_labels_ext = sup_labels.unsqueeze(0).repeat(args.query_num, 1)         #[q, nc*sh]
          sup_labels_ext = torch.transpose(sup_labels_ext, 0, 1)                     #[nc*sh, q]

          test_features_ext = test_features.unsqueeze(0).repeat(n_known*rep_per_class, 1, 1) #[nc*sh, q, 128]
          test_labels_ext = test_labels.unsqueeze(0).repeat(n_known*rep_per_class, 1)        #[nc*sh, q]

          relation_pairs = torch.cat((sup_features_ext, test_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]
          # relarion_labels = torch.zeros(n_known*rep_per_class, args.query_num).to(self.device)
          # relarion_labels = torch.where(
          #   sup_labels_ext!=test_labels_ext,
          #   relarion_labels,
          #   torch.tensor(1.).to(self.device)
          # ).view(-1,1)
        
          ## == Relation Network ===============
          relations = relation_net(relation_pairs).view(-1, n_known*rep_per_class)
          relations_mean = torch.mean(
            torch.stack(
              torch.split(relations, n_known, dim=1),
              dim=2
            ), 2
          )
        
          ## == Open-world Relation-based Acc. ============
          _, ow_predict_labels = torch.max(relations_mean.data, 1)
          ow_total += test_labels.size(0)
          ow_correct += (ow_predict_labels == test_labels).sum().item()

          ## == loss relation ====================
          test_labels_onehot = torch.zeros(
            args.query_num, n_known
          ).to(self.device).scatter_(1, test_labels.view(-1,1), 1)
          loss = self.relation_criterion(relations.data, test_labels_onehot)

        else: # For score approach
          ## == Similarity test ================
          all_sim = cos_similarity(test_features, sup_features)
          # avg_sim = torch.tensor([
          #   torch.mean(all_sim[:, i*rep_per_class:(i+1)*rep_per_class])
          #   for i in known_labels
          # ])
          # prob, predicted_idx = torch.max(avg_sim, 0)
          # predicted_labels = known_labels[predicted_idx].item()
          
          _, predicted_idx = torch.max(all_sim, 1)
          predicted_labels = known_labels[torch.div(predicted_idx, args.n_examplers, rounding_mode='trunc')]
          
          ow_total += test_labels.size(0)
          ow_correct += (predicted_labels == test_labels).sum().item()

        ## == Close World Acc. =================
        _, cw_predict_labels = torch.max(test_outputs, 1)
        cw_total += test_labels.size(0)
        cw_correct += (cw_predict_labels == test_labels).sum().item()
        
        ## == Metric Loss ======================
        loss = self.extractor_criterion(test_outputs, test_labels) # For just CW
        loss = loss.mean()
        total_loss += loss.item()

      total_loss /= len(val_loader)
      total_cw_acc = cw_correct / cw_total
      total_ow_acc = ow_correct / ow_total  

      return total_loss, total_cw_acc, total_ow_acc

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)











  # def train(self,
  #   feature_ext,
  #   relation_net,
  #   batch,
  #   feature_ext_optim, relation_net_optim,
  #   iteration,
  #   args):
    
  #   feature_ext.train()
  #   relation_net.train()

  #   feature_ext_optim.zero_grad()
  #   relation_net_optim.zero_grad()

  #   ### === Prepare data ===============================
  #   support_len = args.ways * args.shot 
  #   support_images, support_labels, query_images, query_labels = batch
  #   support_images = support_images.reshape(-1, *support_images.shape[2:])
  #   support_labels = support_labels.flatten()
  #   query_images = query_images.reshape(-1, *query_images.shape[2:])
  #   query_labels = query_labels.flatten()

  #   # print('support_images: {}'.format(support_images.shape))
  #   # print('support_labels: {}'.format(support_labels))
  #   # print('query_images: {}'.format(query_images.shape))
  #   # print('query_labels: {}'.format(query_labels))

  #   unique_label = torch.unique(support_labels).to(self.device)

  #   support_images = support_images.to(self.device)
  #   support_labels = support_labels.to(self.device)
  #   query_images = query_images.to(self.device)
  #   query_labels = query_labels.to(self.device)

  #   images = torch.cat((support_images, query_images))
  #   labels = torch.cat((support_labels, query_labels))

  #   ### === Feature extractor ==========================
  #   outputs, features = feature_ext.forward(images)
    
  #   ### === Concat features ============================
  #   support_features = features[:support_len] #[w*s, 128]
  #   query_features = features[support_len:]   #[w*q, 128]
  #   support_outputs = outputs[:support_len]

  #   # print('support_features: {}'.format(support_features.shape))
  #   # print('query_features: {}'.format(query_features.shape))

  #   support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num, 1, 1)  #[w*q, w*sh, 128]
  #   support_features_ext = torch.transpose(support_features_ext, 0, 1)                    #[w*sh, w*q, 128]
  #   support_labels_ext = support_labels.unsqueeze(0).repeat(args.ways*args.query_num, 1)      #[w*q, w*sh]
  #   support_labels_ext = torch.transpose(support_labels_ext, 0, 1)                                #[w*sh, w*q]

  #   # print('support_labels: {}'.format(support_labels))
  #   # print('support_labels: {}'.format(support_labels.shape))

  #   query_features_ext = query_features.unsqueeze(0).repeat(args.ways*args.shot, 1, 1) #[w*sh, w*q, 128]
  #   query_labels_ext = query_labels.unsqueeze(0).repeat(args.ways*args.shot, 1)            #[w*sh, w*q]

  #   # print('query_labels: {}'.format(query_labels))
  #   # print('query_labels: {}'.format(query_labels.shape))

  #   relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, args.feature_dim*2) #[w*q*w*sh, 256]
  #   relarion_labels = torch.zeros(args.ways*args.shot, args.ways*args.query_num).to(self.device)
  #   relarion_labels = torch.where(
  #     support_labels_ext!=query_labels_ext,
  #     relarion_labels,
  #     torch.tensor(1.).to(self.device)
  #   ).view(-1,1)

  #   # print(relarion_labels)
  #   # print(relarion_labels.shape)

  #   ### === Relation Network ===========================
  #   relations = relation_net(relation_pairs).view(-1,args.ways) #[w, w*q]
    
  #   ### === Loss & backward ============================
  #   quety_label_pressed = torch.tensor([(unique_label == l).nonzero(as_tuple=True)[0] for l in query_labels], device=self.device)
  #   query_labels_onehot = torch.zeros(
  #     args.ways*args.query_num, args.ways
  #   ).to(self.device).scatter_(1, quety_label_pressed.view(-1,1), 1)
  #   query_labels_onehot = query_labels_onehot.to(self.device)

  #   # print(support_outputs.shape)
  #   # print(support_labels.shape)
  #   loss = self.criterion(
  #     support_outputs,
  #     support_labels,
  #     relations,
  #     query_labels_onehot
  #   )
  #   loss.backward()

  #   torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
  #   torch.nn.utils.clip_grad_norm_(relation_net.parameters(), args.grad_clip)
  #   feature_ext_optim.step()
  #   relation_net_optim.step()

  #   return loss.detach().item()
