import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from pandas import read_csv

from dataset import SimpleDataset
from samplers.relation_sampler import RelationSampler
from trainers.episodic_trainer import train, test


def init_learn(feature_ext, relation, learner, args, device):
  ## == Data ============================
  train_data = read_csv(
    os.path.join(args.data_path, args.train_file),
    sep=',',
    header=None).values
  
  train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])

  print(train_data.shape)
  train_data,\
    val_data = torch.utils.data.random_split(
      train_data,
      [int(train_data.shape[0]*0.9), int(train_data.shape[1]*0.1)]
    )
  
  
  train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
  known_labels = train_dataset.label_set

  sampler = RelationSampler(
    train_dataset,
    n_way=args.ways,
    n_shot=args.shot,
    n_query=args.query_num,
    n_tasks=args.meta_iteration
  )
  train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )
  val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)

  ## == train ===========================
  train(
    feature_ext,
    relation,
    learner,
    train_dataloader,
    val_dataloader,
    known_labels,
    args, device)

  ## == Test ============================
  test()