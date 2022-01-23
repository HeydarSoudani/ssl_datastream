import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import os
import time
from pandas import read_csv

from losses import W_MSE, W_BCE

from dataset import SimpleDataset
from samplers.pt_sampler import PtSampler

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
    # self.criterion = torch.nn.CrossEntropyLoss()
    a = torch.nn.MSELoss()
    self.criterion = W_MSE()
    self.device = device

    # self.prototypes = {
    #   l: torch.zeros(1, args.feature_dim, device=device)
    #   for l in range(args.n_classes)
    # }
  
  def train(self,
    feature_ext,
    relation_net,
    batch,
    feature_ext_optim, relation_net_optim,
    iteration,
    args):
    
    feature_ext.train()
    relation_net.train()

    feature_ext_optim.zero_grad()
    relation_net_optim.zero_grad()

    ### === Prepare data ===============================
    support_len = args.ways * args.shot 
    support_images, support_labels, query_images, query_labels = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    query_images = query_images.reshape(-1, *query_images.shape[2:])
    query_labels = query_labels.flatten()

    # print('support_images: {}'.format(support_images.shape))
    # print('support_labels: {}'.format(support_labels))
    # print('query_images: {}'.format(query_images.shape))
    # print('query_labels: {}'.format(query_labels))

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
    # episode_prototypes = compute_prototypes(
    #   features[:support_len], support_labels
    # )
    # old_prototypes = torch.cat(
    #   [self.prototypes[l.item()] for l in unique_label]
    # )
    # beta = args.beta * iteration / args.meta_iteration
    # new_prototypes = beta * old_prototypes + (1 - beta) * episode_prototypes

    ### === Concat features ============================
    support_features = features[:support_len] #[w*s, 128]
    query_features = features[support_len:]   #[w*q, 128]

    # print('support_features: {}'.format(support_features.shape))
    # print('query_features: {}'.format(query_features.shape))

    support_features_ext = support_features.unsqueeze(0).repeat(args.ways*args.query_num, 1, 1)  #[w*q, w*sh, 128]
    support_features_ext = torch.transpose(support_features_ext, 0, 1)                    #[w*sh, w*q, 128]
    support_labels = support_labels.unsqueeze(0).repeat(args.ways*args.query_num, 1)      #[w*q, w*sh]
    support_labels = torch.transpose(support_labels, 0, 1)                                #[w*sh, w*q]

    # print('support_labels: {}'.format(support_labels))
    # print('support_labels: {}'.format(support_labels.shape))

    query_features_ext = query_features.unsqueeze(0).repeat(args.ways*args.shot, 1, 1) #[w*sh, w*q, 128]
    query_labels_ext = query_labels.unsqueeze(0).repeat(args.ways*args.shot, 1)            #[w*sh, w*q]

    # print('query_labels: {}'.format(query_labels))
    # print('query_labels: {}'.format(query_labels.shape))

    relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, args.feature_dim*2) #[w*q*w*sh, 256]
    relarion_labels = torch.zeros(args.ways*args.shot, args.ways*args.query_num).to(self.device)
    relarion_labels = torch.where(
      support_labels!=query_labels_ext,
      relarion_labels,
      torch.tensor(1.).to(self.device)
    ).view(-1,1)

    # print(relarion_labels)
    # print(relarion_labels.shape)

    ### === Relation Network ===========================
    relations = relation_net(relation_pairs).view(-1,args.ways) #[w, w*q]
    
    ### === Loss & backward ============================
    # loss = self.criterion(outputs[:support_len], support_labels) # without Relation network
    query_labels_onehot = torch.zeros(args.ways*args.query_num, args.ways).to(self.device).scatter_(1, query_labels.view(-1,1), 1)

    print(query_labels.view(-1,1))
    print(query_labels_onehot.view(-1,args.ways))

    query_labels_onehot = query_labels_onehot.to(self.device)

    loss = self.criterion(relations, relarion_labels)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(feature_ext.parameters(), args.grad_clip)
    torch.nn.utils.clip_grad_norm_(relation_net.parameters(), args.grad_clip)
    feature_ext_optim.step()
    relation_net_optim.step()

    return loss.detach().item()


  def evaluate(self,
    feature_ext,
    relation_net,
    val_loader,
    args):

    feature_ext.eval()
    relation_net.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # known_labels = torch.tensor(list(known_labels), device=self.device)
    # pts = torch.cat(
    #   [self.prototypes[l.item()] for l in known_labels]
    # )
    ## == Train data ====================
    train_data = read_csv(os.path.join(args.data_path, args.train_file), sep=',', header=None).values
    train_dataset = SimpleDataset(
      train_data, args,
      transforms=transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426]))
    sampler = PtSampler(
      train_dataset,
      n_way=args.n_classes,
      n_shot=args.shot,
      n_query=0,
      n_tasks=500
    )
    train_dataloader = DataLoader(
      train_dataset,
      batch_sampler=sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=sampler.episodic_collate_fn,
    )
    trainloader = iter(train_dataloader)

    with torch.no_grad():
      total_loss = 0.0
      total = 0.0
      correct = 0.0 

      for i, batch in enumerate(val_loader):
        sup_batch = next(trainloader)
        sup_images, sup_labels, _, _ = sup_batch
        
        sup_images = sup_images.reshape(-1, *sup_images.shape[2:])
        sup_labels = sup_labels.flatten()
        sup_images, sup_labels = sup_images.to(self.device), sup_labels.to(self.device)
        
        samples, labels = batch

        labels = labels.flatten()
        samples, labels = samples.to(self.device), labels.to(self.device)

        _, sup_features = feature_ext.forward(sup_images)
        _, test_features = feature_ext.forward(samples)

        sup_features_ext = sup_features.unsqueeze(0).repeat(args.query_num, 1, 1)  #[q, w*sh, 128]
        sup_features_ext = torch.transpose(sup_features_ext, 0, 1)            #[w*sh, q, 128]
        sup_labels = sup_labels.unsqueeze(0).repeat(args.query_num, 1)        #[q, w*sh]
        sup_labels = torch.transpose(sup_labels, 0, 1)                        #[w*sh, q]

        test_features_ext = test_features.unsqueeze(0).repeat(args.ways*args.shot, 1, 1) #[w*sh, q, 128]
        test_labels = labels.unsqueeze(0).repeat(args.ways*args.shot, 1)            #[w*sh, q]

        relation_pairs = torch.cat((sup_features_ext, test_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]
        relarion_labels = torch.zeros(args.ways*args.shot, args.query_num).to(self.device)
        relarion_labels = torch.where(
          sup_labels!=test_labels,
          relarion_labels,
          torch.tensor(1.).to(self.device)
        ).view(-1,1)
        relations = relation_net(relation_pairs).view(-1, args.ways)

        ## == Relation-based Acc. ============== 
        _,predict_labels = torch.max(relations.data, 1)
        total += labels.size(0)
        correct += (predict_labels == labels).sum().item()

        ## == Cls-based Acc. ===================
        # _, predicted = torch.max(logits, 1)
        # total_cls_acc += labels.size(0)
        # correct_cls_acc += (predicted == labels).sum().item()

        ## == loss =============================
        loss = criterion(relations.data, labels)
        loss = loss.mean()
        total_loss += loss.item()

      total_loss /= len(val_loader)
      total_acc = correct / total  

      return total_loss, total_acc