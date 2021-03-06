import torch
import os
import argparse
import numpy as np
from pandas import read_csv

from model import MyPretrainedResnet50, MLP, weights_init
from dataset import SimpleDataset
from learners.relation_learner import RelationLearner
from learners.pt_learner import PtLearner
from detectors.similarity_detector import SimDetector
from utils.memory import OperationalMemory
from losses.metric_loss import TotalLoss as TotalMetricLoss
from losses.pt_loss import TotalLoss as TotalPtLoss
from phases.batch_learn import batch_learn
from phases.init_learn import init_learn
from phases.stream_learn import stream_learn
from phases.zeroshot_test import zeroshot_test

from visualize import visualization, set_novel_label

## == Params ========================
parser = argparse.ArgumentParser()

parser.add_argument(
  '--phase',
  type=str,
  choices=[
    'batch_learn',
    'init_learn',
    'zeroshot_test',
    'stream_learn',
    'zeroshot_test_base',
    'visualization'
  ],
  default='plot',
  help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'fmnist',
    'cifar10',
    'cifar100',
  ],
  default='cifar10',
  help=''
)
parser.add_argument(
  '--which_model',
  type=str,
  choices=['best', 'last'],
  default='best',
  help='')
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=2, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--meta_iteration', type=int, default=3000, help='')
parser.add_argument('--log_interval', type=int, default=5, help='must be less then meta_iteration parameter')
parser.add_argument('--relation_train_interval', type=int, default=5, help='must be less then meta_iteration parameter')

# retrain
parser.add_argument('--buffer_size', type=int, default=1000, help='')
parser.add_argument('--retrain_epochs', type=int, default=1, help='')
parser.add_argument('--retrain_meta_iteration', type=int, default=1000, help='')
parser.add_argument('--known_retrain_interval', type=int, default=10000, help='')
parser.add_argument('--known_per_class', type=int, default=100, help='for known buffer')

# Sampler
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=5, help='')
parser.add_argument('--query_num', type=int, default=5, help='')

# Optimizer
parser.add_argument('--lr_ext', type=float, default=0.001, help='')
parser.add_argument('--lr_rel', type=float, default=0.001, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=0.0005, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=5.0)

# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.5)

# Model
parser.add_argument('--feature_dim', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument(
  '--rep_approach',
  type=str,
  default='prototype',
  choices=['exampler, prototype'],
  help='representation approach to show known classes'
)
parser.add_argument('--n_examplers', type=int, default=2)
parser.add_argument(
  '--similarity_approach',
  type=str,
  default='score',
  choices=['score, relation'],
  help='similarity approach to compare two representators'
)

# Transform
parser.add_argument('--use_transform', action='store_true')

# Loss function
parser.add_argument("--lambda_1", type=float, default=1.0, help="Metric Coefficien in loss function")
parser.add_argument("--lambda_2", type=float, default=1.0, help="relation Coefficient in loss function")
parser.add_argument("--temp_scale", type=float, default=0.2, help="Temperature scale for DCE in loss function")

# memory
parser.add_argument('--mem_sel_type', type=str, default='fixed_mem', choices=['fixed_mem', 'pre_class'], help='')
parser.add_argument('--mem_total_size', type=int, default=2000, help='')
parser.add_argument('--mem_per_class', type=int, default=100, help='')
parser.add_argument('--mem_sel_method', type=str, default='rand', choices=['rand', 'soft_rand'], help='')
parser.add_argument('--mem_novel_acceptance', type=int, default=150, help='')

# Device and Randomness
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')

# Save and load model
parser.add_argument('--save', type=str, default='saved/', help='')

args = parser.parse_args()

## == Set class number =================
if args.dataset in ['mnist', 'fmnist', 'cifar10']:
  args.n_classes = 10
elif args.dataset in ['cifar100']:
  args.n_classes = 100

## == Device ===========================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Device: {}'.format(device))

## == Apply seed =======================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Define & Load (Metric) learner ===
# extractor_criterion = TotalMetricLoss(args)
# relation_criterion = torch.nn.MSELoss()
# learner = RelationLearner(extractor_criterion, relation_criterion, device, args)
# if args.phase in ['zeroshot_test', 'stream_learn', 'visualization']:
#   learner_path = os.path.join(args.save, "learner.pt")
#   learner.load(learner_path)
#   print("Load learner from {}".format(learner_path))

## == Define & Load (Pt) learner ========
criterion = TotalPtLoss(device, args)
learner = PtLearner(criterion, device, args)
if args.phase in ['zeroshot_test', 'stream_learn', 'visualization']:
  learner_path = os.path.join(args.save, "learner.pt")
  learner.load(learner_path)
  print("Load learner from {}".format(learner_path))

## == Define & Load learner =============
detector = SimDetector(device)
if args.phase in ['zeroshot_test', 'stream_learn', 'visualization']:
  detector_path = os.path.join(args.save, "detector.pt")
  detector.load(detector_path)
  print("Load detector from {}".format(detector_path))

## == Define Memory =====================
memory = OperationalMemory(device,
                            selection_type=args.mem_sel_type,
                            total_size=args.mem_total_size,
                            per_class=args.mem_per_class,
                            novel_acceptance=args.mem_novel_acceptance)
if args.phase in ['zeroshot_test', 'stream_learn', 'visualization']:
  memory_path = os.path.join(args.save, "memory.pt")
  memory.load(memory_path)
  print("Load memory from {}".format(memory_path))

## == Define Feature extractor & Relation network ==
feature_ext = MyPretrainedResnet50(args)
relation_net = MLP(args)
# feature_ext.apply(weights_init)
# relation_net.apply(weights_init)
feature_ext.to(device)
relation_net.to(device)
# print(feature_ext)

## == Load Feature extractor & Relation network ==
if args.phase in ['zeroshot_test', 'stream_learn', 'visualization']:
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

# == Print feature_ext layers and params ====
total_params = sum(p.numel() for p in feature_ext.parameters())
total_params_trainable = sum(p.numel() for p in feature_ext.parameters() if p.requires_grad)
print('Total params: {}'.format(total_params))
print('Total trainable params: {}'.format(total_params_trainable))

## == load data ========================
args.data_path = 'data/'
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)
args.stream_file = '{}_stream.csv'.format(args.dataset)

train_data = read_csv(
  os.path.join(args.data_path, args.train_file),
  sep=',',
  header=None).values

## == Get base labels ==================
base_labels = SimpleDataset(train_data, args).label_set

## == training =========================
if __name__ == '__main__':
  ## == Batch learning ===
  if args.phase == 'batch_learn':
    batch_learn(feature_ext, args, device)
  
  ## == Data Stream ======
  elif args.phase == 'init_learn':
    init_learn(
      feature_ext,
      relation_net,
      learner,
      detector,
      memory,
      train_data,
      base_labels,
      args, device
    )
  elif args.phase == 'stream_learn':
    stream_learn(
      feature_ext,
      relation_net,
      learner,
      detector,
      memory,
      base_labels,
      args, device
    )
  elif args.phase == 'zeroshot_test':
    zeroshot_test(
      feature_ext,
      relation_net,
      learner,
      detector,
      base_labels,
      args, device
    )
  ## == visualization ===== 
  elif args.phase == 'visualization':
    stream_data = set_novel_label(base_labels, args)
    # stream_data = read_csv(
    #   os.path.join(args.data_path, args.stream_file),
    #   sep=',',
    #   header=None).values
    stream_dataset = SimpleDataset(stream_data, args)
    visualization(feature_ext, stream_dataset, device)











