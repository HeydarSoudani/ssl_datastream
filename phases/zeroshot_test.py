import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from pandas import read_csv
from losses import cos_similarity

from dataset import SimpleDataset

def zeroshot_test(feature_ext,
                  relation_net,
                  learner,
                  known_labels,
                  args, device):
  print('================================ Zero-Shot Test ================================')
  feature_ext.eval()
  relation_net.eval()
  
  ## == Load stream data ==============================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values
  
  stream_batch = 1
  stream_dataset = SimpleDataset(stream_data, args)
  streamloader = DataLoader(dataset=stream_dataset, batch_size=stream_batch, shuffle=False)

  
  ## == Create prototypes and known_labels ============
  n_known = len(known_labels)
  pt_per_class = 1
  known_labels = torch.tensor(list(known_labels), device=device)
  print('Known labels: {}'.format(known_labels))
  
  if args.rep_approach == 'prototype':
    sup_features = torch.cat(
      [learner.prototypes[l.item()] for l in known_labels]
    )
  elif args.rep_approach == 'exampler':
    sup_features = torch.cat(
      [learner.examplers[l.item()] for l in known_labels]
    )
  sup_labels = known_labels

  ## == 
  with torch.no_grad():
    for i, batch in enumerate(streamloader):

      if i < 20:
        # Query set
        test_image, test_label = batch
        test_label = test_label.flatten()
        test_image, test_label = test_image.to(device), test_label.to(device)
        real_novelty = test_label.item() not in known_labels
        _, test_feature = feature_ext.forward(test_image)

        # ## == Relation Network preparation =====
        # sup_features_ext = sup_features.unsqueeze(0).repeat(stream_batch, 1, 1)  #[q, w*sh, 128]
        # sup_features_ext = torch.transpose(sup_features_ext, 0, 1)               #[w*sh, q, 128]
        # sup_labels_ext = sup_labels.unsqueeze(0).repeat(stream_batch, 1)             #[q, w*sh]
        # sup_labels_ext = torch.transpose(sup_labels_ext, 0, 1)
        #                            #[w*sh, q]
        # test_features_ext = test_features.unsqueeze(0).repeat(n_known*pt_per_class, 1, 1) #[w*sh, q, 128]
        # test_labels_ext = test_labels.unsqueeze(0).repeat(n_known*pt_per_class, 1)        #[w*sh, q]
        
        # relation_pairs = torch.cat((sup_features_ext, test_features_ext), 2).view(-1, args.feature_dim*2) #[q*w*sh, 256]

        # ## == Relation Network ===================
        # relations = relation_net(relation_pairs).view(-1, args.ways)
        # prob, predict_labels = torch.max(relations.data, 1)
        
        ## == Similarity score ==================
        print(test_feature)
        print(sup_features)
        all_sim = cos_similarity(test_feature, sup_features)
        prob, predict_label = torch.max(all_sim, 1)
        # if (i+1) % 1000 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %s, %s"%(
          i+1, test_label.item(), predict_label, prob, real_novelty,
          tuple(np.around(np.array(all_sim.tolist()),2)[0])
        ))
    
