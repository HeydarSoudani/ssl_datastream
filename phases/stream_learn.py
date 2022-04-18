import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from pandas import read_csv

from dataset import SimpleDataset
from evaluation import in_stream_evaluation
from trainers.episodic_trainer import train


def stream_learn(feature_ext,
                relation_net,
                learner,
                detector,
                memory,
                base_labels,
                args, device
  ):
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
  n_known = len(base_labels)
  known_labels = torch.tensor(list(base_labels), device=device)
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
        known_buffer[predicted_label].append((test_image, test_label.item()))

      if (i+1) % 100 == 0:
        print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s, %d" %
          (i+1, test_label.item(), predicted_label, prob, real_novelty, detected_novelty, len(unknown_buffer)))


    if (i+1) % args.known_retrain_interval == 0 \
      or len(unknown_buffer) == args.buffer_size:
      print('=== Retraining... =================')

      # == Preparing buffer ==================
      if (i+1) % args.known_retrain_interval == 0:
        buffer = []
        for label, data in known_buffer.items():
          n = len(data)
          if n > args.known_per_class:
            idxs = np.random.choice(
              range(n), size=args.known_per_class, replace=False)
            buffer.extend([data[i] for i in idxs])
          else:
            buffer.extend(data)

      elif len(unknown_buffer) == args.buffer_size:
        buffer = unknown_buffer

      # == 1) evaluation ======================
      sample_num = i-last_idx
      CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
        detection_results, detector._known_labels)

      print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
            (sample_num, CwCA, M_new, F_new))
      print("confusion matrix: \n%s\n" % cm)
      print("acc per class: %s\n" % acc_per_class)
      f.write("[In sample %2d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
              (i, sample_num, CwCA, M_new, F_new))
      f.write("acc per class: %s\n" % acc_per_class)

      # == 2) Preparing retrain data ==========
      new_train_data = memory.select(buffer, return_data=True)
      print('Retrain data number: {}'.format(new_train_data.shape[0]))
      print('='*30)

      # == 3) Retraining Model ================
      train(
        feature_ext,
        relation_net,
        learner,
        new_train_data,
        args
      )
      new_known_labels = set(int(new_train_data[:, -1]))

      # == 4) Recalculating Detector ==========
      detector.threshold_calculation(
        new_train_data,
        feature_ext,
        representors,
        new_known_labels,
        args
      )
      print("Detector Threshold: {}".format(detector.thresholds))  
      detector_path = os.path.join(args.save, "detector.pt") 
      detector.save(detector_path)
      print("Detector has been saved in {}.".format(detector_path))

      # == 5) Update parameters ===============
      known_labels = list(known_buffer.keys())
      labels_diff = list(set(new_known_labels)-set(known_labels))
      for label in labels_diff:
        print('Class {} detected at {}'.format(label, i))
        f.write("[Class %2d], Detected point: %5d \n" % (label, i))

      if len(unknown_buffer) == args.buffer_size:
        if len(labels_diff) != 0:
          for label in labels_diff:
            known_buffer[label] = []
        unknown_buffer.clear()
      if (i+1) % args.known_retrain_interval == 0:
        known_buffer = {i: [] for i in detector._known_labels}

      # == Set parameters =====
      detection_results.clear()
      last_idx = i

      print('=== Streaming... =================')

  # == Last evaluation ========================
  sample_num = i-last_idx
  CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
      detection_results, detector._known_labels)
  print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
        (sample_num, CwCA, M_new, F_new))
  print("confusion matrix: \n%s" % cm)
  print("acc per class: %s\n" % acc_per_class)
  f.write("[In sample %5d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
          (i, sample_num, CwCA, M_new, F_new))
  f.write("acc per class: %s\n" % acc_per_class)
  f.close()

