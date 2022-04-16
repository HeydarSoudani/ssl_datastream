import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from pandas import read_csv

from dataset import SimpleDataset
from utils.preparation import transforms_preparation
from evaluation import in_stream_evaluation


def stream_learn(feature_ext,
                  relation_net,
                  learner,
                  detector,
                 args, device):
  print('================================ Stream Learing ================================')
  ## == Set retrain params ====================
  args.epochs = args.retrain_epochs
  args.meta_iteration = args.retrain_meta_iteration

  ## == Data ==================================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values
  stream_batch = 1
  stream_dataset = SimpleDataset(stream_data, args)
  streamloader = DataLoader(dataset=stream_dataset, batch_size=stream_batch, shuffle=False)

  ## == Classes start points ===================
  f = open('output.txt', 'w')
  all_labels = stream_dataset.labels
  label_set = stream_dataset.label_set
  for label in label_set:
    start_point = np.where(all_labels == label)[0][0]
    print('Class {} starts at {}'.format(label, start_point))
    f.write("[Class %5d], Start point: %5d \n" % (label, start_point))

  ## == Define representors and known_labels ============
  n_known = len(known_labels_set)
  known_labels = torch.tensor(list(known_labels_set), device=device)
  print('Known labels: {}'.format(known_labels))
  
  if args.rep_approach == 'prototype':
    rep_per_class = 1
    representors = torch.cat(
      [learner.prototypes[l.item()] for l in known_labels]
    )
  elif args.rep_approach == 'exampler':
    rep_per_class = args.n_examplers
    representors = torch.cat(
      [learner.examplers[l.item()] for l in known_labels]
    )
  sup_labels = known_labels

  # == Stream ================================
  unknown_buffer = []
  known_buffer = {i: [] for i in detector._known_labels}
  detection_results = []
  last_idx = 0

  for i, batch in enumerate(streamloader):
    feature_ext.eval()
    relation_net.eval()
    
    with torch.no_grad():
      test_image, test_label = batch
      test_label = test_label.flatten()
      test_image, test_label = test_image.to(device), test_label.to(device)
      real_novelty = test_label.item() not in known_labels
      _, test_feature = feature_ext.forward(test_image)

      detected_novelty, predicted_label, prob, _ = detector(test_feature, representors, rep_per_class)  
      detection_results.append((test_label.item(), predicted_label, real_novelty, detected_novelty))
        
      test_image = torch.squeeze(test_image, 0)  # [1, 28, 28]
      if detected_novelty:
        unknown_buffer.append((test_image, test_label))
      else:
        known_buffer[predicted_label].append((test_image, test_label))

      if (i+1) % 100 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s, %d" %
          (i+1, test_label.item(), predicted_label, prob, real_novelty, detected_novelty, len(unknown_buffer)))






