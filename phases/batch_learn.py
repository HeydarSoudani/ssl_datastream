import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from pandas import read_csv

from dataset import SimpleDataset
from trainers.batch_trainer import train, test


def batch_learn(model, args, device):
  print('================================ Batch Learning =========================')
  
  ## == data ===============================
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',',
    header=None).values
  # test_data = read_csv(
  #   os.path.join(args.data_path, args.test_file),
  #   sep=',',
  #   header=None).values

  train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])

  # test_transform = transforms.Compose([
  #   transforms.ToPILImage(),
  #   transforms.ToTensor(),
  #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  # ])

  train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [5500, 500])
  # test_dataset = SimpleDataset(test_data, args, transforms=test_transform)

  train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
  val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)
  # test_dataloader = DataLoader(dataset=test_dataset,
  #                               batch_size=args.batch_size,
  #                               shuffle=False)

  ## == train ===========================
  train(model, train_dataloader, val_dataloader, args, device)

  ## == Test model ======================
  # print('Test with last model')
  # test(model, test_dataloader, args, device)

  # print('Test with best model')
  # try: model.load_state_dict(torch.load(args.best_model_path), strict=False)
  # except FileNotFoundError: pass
  # else: print("Load model from {}".format(args.best_model_path))
  # test(model, test_dataloader, args, device)


