import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import time
from pandas import read_csv

from dataset import SimpleDataset


def zeroshot_test(feature_ext,
                  relation_net,
                  detector,
                  args,
                  device,
                  known_labels=None):
  print('================================ Zero-Shot Test ================================')
  feature_ext.eval()
  relation_net.eval()

  ## == Similarity score ==============================
  cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
  
  ## == Load stream data ==============================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values
  
  stream_dataset = SimpleDataset(
    stream_data, args,
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
  streamloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

  ## == Load train data ============================== 
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',', header=None).values

  train_dataset = SimpleDataset(
    train_data, args,
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
  sampler = PtSampler(
    train_dataset,
    n_way=args.n_classes,
    n_shot=args.shot,
    n_query=0,
    n_tasks=len(dataloader)
  )
  train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )
  trainloader = iter(train_dataloader)

  ## == Load models ===================================
  if args.which_model == 'best':
    feature_ext_path = os.path.join(args.save, "feature_ext_best.pt")
    relation_net_path = os.path.join(args.save, "relation_net_best.pt")
    try:
      feature_ext.load_state_dict(torch.load(feature_ext_path))
      relation_net.load_state_dict(torch.load(relation_net_path))
    except FileNotFoundError: pass
    else:
      print("Load feature_ext from {} and relation_net from {}".format(feature_ext_path, relation_net_path))
  
  elif args.which_model == 'last':
    feature_ext_path = os.path.join(args.save, "feature_ext_last.pt")
    relation_net_path = os.path.join(args.save, "relation_net_last.pt")
    try:
      feature_ext.load_state_dict(torch.load(feature_ext_path))
      relation_net.load_state_dict(torch.load(relation_net_path))
    except FileNotFoundError: pass
    else:
      print("Load feature_ext from {} and relation_net from {}".format(feature_ext_path, relation_net_path))

  ## == 
  with torch.no_grad():
    for i, batch in enumerate(streamloader):
      
      # Support set
      sup_batch = next(trainloader)
      sup_images, sup_labels, _, _ = sup_batch
      sup_images = sup_images.reshape(-1, *sup_images.shape[2:])
      sup_labels = sup_labels.flatten()
      sup_images, sup_labels = sup_images.to(self.device), sup_labels.to(self.device)

      # Query set
      test_images, test_labels = batch
      test_labels = test_labels.flatten()
      test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)

      _, sup_features = feature_ext.forward(sup_images)
      _, test_features = feature_ext.forward(test_images)

      ## == Relation Network preparation =====
      sup_features_ext = sup_features.unsqueeze(0).repeat(args.query_num, 1, 1)  #[q, nc*sh, 128]
      sup_features_ext = torch.transpose(sup_features_ext, 0, 1)                 #[nc*sh, q, 128]
      sup_labels = sup_labels.unsqueeze(0).repeat(args.query_num, 1)             #[q, nc*sh]
      sup_labels = torch.transpose(sup_labels, 0, 1)                             #[nc*sh, q]
      test_features_ext = test_features.unsqueeze(0).repeat(args.n_classes*args.shot, 1, 1) #[nc*sh, q, 128]
      test_labels_ext = test_labels.unsqueeze(0).repeat(args.n_classes*args.shot, 1)        #[nc*sh, q]
      relation_pairs = torch.cat((sup_features_ext, test_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]

      ## == Similarity score ==================
      feature1, features2 = torch.split(relation_pairs, 2, dim=1)
      sim_score = cos_sim(feature1, features2).view(-1, args.n_classes)

      print(test_labels)
      print(sim_score)
      time.sleep(2)




      


