from torch.utils.data import DataLoader
import os
from pandas import read_csv

from dataset import SimpleDataset
from utils.preparation import dataloader_preparation, transforms_preparation
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
  train_data,
  known_labels,
  args, device
):
  ## == train ===========================
  train(
    feature_ext,
    relation_net,
    learner,
    train_data,
    args, device)

  ## == Test ============================
  # init_test(
  #   feature_ext,
  #   relation_net,
  #   learner,
  #   known_labels,
  #   args)