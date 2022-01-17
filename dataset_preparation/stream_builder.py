import pandas as pd 
import numpy as np
import argparse
import random
import gzip
import os

def load_mnist(path, kind='train'):
  """Load MNIST data from `path`"""
  labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
  images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

  with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
  with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

  return images, labels


## == Params =====================
parser = argparse.ArgumentParser()
parser.add_argument('--class_num', type=int, default=10, help='')
parser.add_argument('--seen_class_num', type=int, default=5, help='')
parser.add_argument('--spc', type=int, default=1200, help='samples per class')
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]
parser.add_argument('--saved', type=str, default='./data/', help='')
parser.add_argument('--seed', type=int, default=2, help='')
args = parser.parse_args()

# = Add some variables to args ===
args.data_path = 'data/{}'.format(args.dataset)
args.train_file = '{}_train.csv'.format(args.dataset)
args.stream_file = '{}_stream.csv'.format(args.dataset)


## == Apply seed =================
np.random.seed(args.seed)


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
    # X_train, y_train = load_mnist(path, kind='train') #(60000, 784), (60000,)
    # X_test, y_test = load_mnist(path, kind='t10k')    #(10000, 784), (10000,)
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

  data = np.concatenate((X_train, X_test), axis=0)  #(70000, 784)
  labels = np.concatenate((y_train, y_test), axis=0)#(70000,)

  # == split data by classes =================
  class_data = {}
  for class_label in set(labels):
    class_data[class_label] = []  
  for idx, sample in enumerate(data):
    class_data[labels[idx]].append(sample)

  # == Select seen & unseen classes ==========
  seen_class = np.random.choice(args.class_num, args.seen_class_num, replace=False)
  unseen_class = [x for x in list(set(labels)) if x not in seen_class]
  # seen_class = [0, 1, 2, 3, 4] 
  # unseen_class = [5, 6, 7, 8, 9]
  print('seen: {}'.format(seen_class))
  print('unseen: {}'.format(unseen_class))

  # == Preparing train dataset and test seen data ===
  train_data = []
  test_data_seen = []
  for seen_class_item in seen_class:
    seen_data = np.array(class_data[seen_class_item])

    del class_data[seen_class_item]

    np.random.shuffle(seen_data)
    last_idx = args.spc
    test_data_length = seen_data.shape[0] - last_idx
    train_part = seen_data[:last_idx]
    test_part = seen_data[last_idx:]

    train_data_class = np.concatenate((train_part, np.full((last_idx , 1), seen_class_item)), axis=1)
    train_data.extend(train_data_class)

    test_data_class = np.concatenate((test_part, np.full((test_data_length , 1), seen_class_item)), axis=1)
    test_data_seen.extend(test_data_class)
  
  train_data = np.array(train_data) #(6000, 785)
  np.random.shuffle(train_data)
  
  pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
    header=None,
    index=None
  )
  print('train data saved in {}'.format(os.path.join(args.saved, args.train_file)))

  # == Preparing test(stream) dataset ================
  test_data_seen = np.array(test_data_seen) #(30000, 785)
  all_temp_data = []
  add_class_point = 6000
  
  np.random.shuffle(test_data_seen)
  all_temp_data = test_data_seen[:add_class_point]
  test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

  while True:

    if len(unseen_class) != 0:
      rnd_uns_class = unseen_class[0]
      unseen_class.remove(rnd_uns_class)
      
      selected_data = np.array(class_data[rnd_uns_class])
      del class_data[rnd_uns_class]
      temp_data_with_label = np.concatenate((selected_data, np.full((selected_data.shape[0] , 1), rnd_uns_class)), axis=1)
      test_data_seen = np.concatenate((test_data_seen, temp_data_with_label), axis=0)

    np.random.shuffle(test_data_seen)
    all_temp_data = np.concatenate((all_temp_data, test_data_seen[:add_class_point]), axis=0)
    # all_temp_data.append(test_data_seen[:add_class_point])
    test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

    if len(unseen_class) == 0:
      np.random.shuffle(test_data_seen)
      all_temp_data = np.concatenate((all_temp_data, test_data_seen), axis=0)
      break
  
  test_data = np.stack(all_temp_data)
  pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.stream_file),
    header=None,
    index=None
  )
  print('stream data saved in {}'.format(os.path.join(args.saved, args.stream_file)))
  # dataset = np.concatenate((train_data, test_data), axis=0)
  # file_path = './dataset/fashion-mnist_stream.csv'
  # pd.DataFrame(dataset).to_csv(file_path, header=None, index=None)
