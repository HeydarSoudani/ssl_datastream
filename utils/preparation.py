import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from dataset import SimpleDataset
from samplers.relation_sampler import RelationSampler

def dataloader_preparation(train_data, val_data, args):
  if val_data == []:
    train_data, val_data = train_test_split(train_data, test_size=0.1)
  
  if args.use_transform:
    train_transform, test_transform = transforms_preparation()
    train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
    val_dataset = SimpleDataset(val_data, args, transforms=test_transform)
  else:
    train_dataset = SimpleDataset(train_data, args)
    val_dataset = SimpleDataset(val_data, args)

  known_labels = train_dataset.label_set
  print('Known labels: {}'.format(known_labels))

  # sampler = PtSampler(
  #   train_dataset,
  #   n_way=args.ways,
  #   n_shot=args.shot,
  #   n_query=args.query_num,
  #   n_tasks=args.meta_iteration
  # )
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
  val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.query_num,
    drop_last=True,
    shuffle=False
  )
  return train_dataloader, val_dataloader, known_labels

def test_dataloader_preparation(data, args):
  _, test_transform = transforms_preparation()
  val_dataset = SimpleDataset(data, args, transforms=test_transform)

def transforms_preparation():
  train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    # CIFAR10Policy(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Cutout(n_holes=1, length=16),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426]),
    # transforms.RandomErasing(probability=args.p, sh=args.sh, r1=args.r1, mean=[0.5, 0.5, 0.5]),
  ])

  test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  return train_transform, test_transform

def memory_samples_preparation(data, device, args):
  dataset = SimpleDataset(data, args)
  dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
  samples = []

  for i, data in enumerate(dataloader):
    sample, label = data
    sample, label = sample.to(device), label.to(device)
    samples.append((torch.squeeze(sample, 0).detach(), label.item())) #[1, 28, 28]))
  
  return samples
      