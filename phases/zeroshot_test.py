import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
import time
from pandas import read_csv

from dataset import SimpleDataset
from samplers.pt_sampler import PtSampler

def zeroshot_test(feature_ext,
                  relation_net,
                  learner,
                  known_labels,
                  args, device):
  print('================================ Zero-Shot Test ================================')
  feature_ext.eval()
  relation_net.eval()

  ## == Similarity score ==============================
  cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
  
  ## == Load stream data ==============================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values
  
  stream_batch = 1
  stream_dataset = SimpleDataset(stream_data, args)
  streamloader = DataLoader(dataset=stream_dataset, batch_size=stream_batch, shuffle=False)

  ## == Load models ===================================
  if args.which_model == 'best':
    feature_ext_path = os.path.join(args.save, "feature_ext_best.pt")
    relation_net_path = os.path.join(args.save, "relation_net_best.pt")
    try:
      feature_ext.load_state_dict(torch.load(feature_ext_path))
      relation_net.load_state_dict(torch.load(relation_net_path))
    except FileNotFoundError: pass
    else:
      print("Load feature_ext from {}".format(feature_ext_path))
      print("Load relation_net from {}".format(relation_net_path))
  
  elif args.which_model == 'last':
    feature_ext_path = os.path.join(args.save, "feature_ext_last.pt")
    relation_net_path = os.path.join(args.save, "relation_net_last.pt")
    try:
      feature_ext.load_state_dict(torch.load(feature_ext_path))
      relation_net.load_state_dict(torch.load(relation_net_path))
    except FileNotFoundError: pass
    else:
      print("Load feature_ext from {}".format(feature_ext_path))
      print("Load relation_net from {}".format(relation_net_path))
  ##
  known_labels = torch.tensor(list(known_labels), device=device)
  print('Known labels: {}'.format(known_labels))
  pts = torch.cat(
    [learner.prototypes[l.item()] for l in known_labels]
  )

  ## == 
  with torch.no_grad():
    for i, batch in enumerate(streamloader):

      if i < 2000:
        # Suppoer set
        sup_features = pts # if use prototypes
        sup_labels = known_labels

        # Query set
        test_images, test_labels = batch
        test_labels = test_labels.flatten()
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        _, test_features = feature_ext.forward(test_images)

        ## == Relation Network preparation =====
        sup_features_ext = sup_features.unsqueeze(0).repeat(stream_batch, 1, 1)  #[q, w*sh, 128]
        sup_features_ext = torch.transpose(sup_features_ext, 0, 1)               #[w*sh, q, 128]
        sup_labels = sup_labels.unsqueeze(0).repeat(stream_batch, 1)             #[q, w*sh]
        sup_labels = torch.transpose(sup_labels, 0, 1)                           #[w*sh, q]
        test_features_ext = test_features.unsqueeze(0).repeat(args.ways*args.shot, 1, 1) #[w*sh, q, 128]
        test_labels_ext = test_labels.unsqueeze(0).repeat(args.ways*args.shot, 1)        #[w*sh, q]
        
        relation_pairs = torch.cat((sup_features_ext, test_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]

        ## == Relation Network ===================
        relations = relation_net(relation_pairs).view(-1, args.ways)
        _, predict_labels = torch.max(relations.data, 1)
        

        ## == Similarity score ==================
        # print(torch.split(relation_pairs, 2, dim=1))
        # print(len(torch.split(relation_pairs, args.feature_dim, dim=1)))
        # feature1, features2 = torch.split(relation_pairs, args.feature_dim, dim=1)
        # sim_score = cos_sim(feature1, features2).view(-1, args.ways)
        # _,predict_labels = torch.max(sim_score, 1)
        # predict_labels = known_labels[predict_labels]
        # print("true label: {}".format(test_labels.data))
        # print("predict label: {}".format(predict_labels.data))
        # print(sim_score.data)

        print("[stream %5d]: %d, %2d"%
          (i+1, test_labels.item(), predict_labels))
        print(relations.data.item())
    



      


