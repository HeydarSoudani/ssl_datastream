import pandas as pd 
import numpy as np
import argparse
import os

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]
parser.add_argument('--seed', type=int, default=2, help='')
parser.add_argument('--saved', type=str, default='./data/', help='')
args = parser.parse_args()

# = Add some variables to args ===
args.data_path = 'data/{}'.format(args.dataset)
args.train_file = '{}_train.csv'.format(args.dataset)
args.test_file = '{}_test.csv'.format(args.dataset)


## == Apply seed ======================
np.random.seed(args.seed)


## == Save dir ========================
if not os.path.exists(args.saved):
  os.makedirs(args.saved)


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset == 'mnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  if args.dataset == 'fmnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Cifar10 dataset ==================
  if args.dataset == 'cifar10':
    train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
    test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
  ## ========================================
  ## ========================================

  train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
  test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)


  pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
    header=None,
    index=None
  )
  pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.test_file),
    header=None,
    index=None
  )
  
