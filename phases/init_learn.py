from torch.utils.data import DataLoader
import os
from pandas import read_csv

from dataset import SimpleDataset
from utils.preparation import transforms_preparation
from trainers.episodic_trainer import train


def init_test(
  feature_ext,
  relation_net,
  learner,
  known_labels,
  args
):
  _, test_transform = transforms_preparation()
  test_data = read_csv(
    os.path.join(args.data_path, args.test_file),
    sep=',',
    header=None).values
  test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

  print('Test with last model ...')
  _, cw_acc, ow_acc = learner.evaluate(feature_ext,
                                        relation_net,
                                        test_dataloader,
                                        known_labels,
                                        args)
  print('CW: {}, OW: {}'.format(cw_acc, ow_acc))

  print('Test with best model ...')
  try:
    feature_ext.load(os.path.join(args.save, "feature_ext_best.pt"))
  except FileNotFoundError: pass
  else:
    print("Load model from {}".format(
      os.path.join(args.save, "feature_ext_best.pt")
    ))
  _, cw_acc, ow_acc = learner.evaluate(feature_ext,
                                        relation_net,
                                        test_dataloader,
                                        known_labels,
                                        args)
  print('CW: {}, OW: {}'.format(cw_acc, ow_acc))

def init_learn(
  feature_ext,
  relation_net,
  learner,
  detector,
  train_data,
  known_labels,
  args, device
):
  ## == Episodic train ===========
  train(
    feature_ext,
    relation_net,
    learner,
    train_data,
    args)

   
  ## == Calculate theresholds ====
  if args.rep_approach == 'prototype':
    representors = learner.prototypes
  elif args.rep_approach == 'exampler':
    representors = learner.examplers
  
  detector.threshold_calculation(
    train_data,
    feature_ext,
    representors,
    known_labels,
    args
  )
  detector_path = os.path.join(args.save, "detector.pt") 
  detector.save(detector_path)
  print("Detector has been saved in {}.".format(detector_path))
  
  ## == Test =====================
  # init_test(
  #   feature_ext,
  #   relation_net,
  #   learner,
  #   known_labels,
  #   args)