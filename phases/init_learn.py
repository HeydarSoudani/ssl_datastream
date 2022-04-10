import torch
from torch.utils.data import DataLoader
import os
from pandas import read_csv

from dataset import SimpleDataset
from utils.preparation import dataloader_preparation, transforms_preparation
from trainers.episodic_trainer import train


def init_test(feature_ext, learner, known_labels, args):
  _, test_transform = transforms_preparation()
  test_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',',
    header=None).values
  test_dataset = SimpleDataset(test_data, args, transforms=test_transform)
  test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

  print('Test with last model ...')
  _, acc_dis, acc_cls = learner.evaluate(feature_ext,
                                        test_dataloader,
                                        known_labels,
                                        args)
  print('Dist: {}, Cls: {}'.format(acc_dis, acc_cls))

  print('Test with best model ...')
  try: feature_ext.load_state_dict(torch.load(args.best_model_path), strict=False)
  except FileNotFoundError: pass
  else: print("Load model from {}".format(args.best_model_path))
  _, acc_dis, acc_cls = learner.evaluate(feature_ext,
                                        test_dataloader,
                                        known_labels,
                                        args)
  print('Dist: {}, Cls: {}'.format(acc_dis, acc_cls))
  


def init_learn(
  feature_ext,
  relation_net,
  learner,
  train_data,
  args, device
):
  ## == Data ============================
  train_dataloader,\
    val_dataloader,\
      known_labels = dataloader_preparation(train_data, [], args)
  

  ## == train ===========================
  train(
    feature_ext,
    relation_net,
    learner,
    train_dataloader,
    val_dataloader,
    known_labels,
    args, device)

  ## == Test ============================
  init_test(feature_ext, learner, known_labels, args)