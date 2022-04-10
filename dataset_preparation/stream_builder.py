import pandas as pd 
import numpy as np
import argparse
import random
import gzip
import time
import os


if __name__ == '__main__':

  ## == Params ===========================
  parser = argparse.ArgumentParser()
  parser.add_argument('--seen_class_num', type=int, default=5, help='')
  parser.add_argument('--spc', type=int, default=1200, help='samples per class for initial dataset')
  parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]
  parser.add_argument('--saved', type=str, default='./data/', help='')
  parser.add_argument('--seed', type=int, default=39, help='')  # seed=1 for regular novel class selection
  args = parser.parse_args()

  # = Add some variables to args =========
  args.data_path = 'data/{}'.format(args.dataset)
  args.train_file = '{}_train.csv'.format(args.dataset)
  args.stream_file = '{}_stream.csv'.format(args.dataset)

  ## == Apply seed =======================
  np.random.seed(args.seed)

  ## == Set class number =================
  if args.dataset in ['mnist', 'fmnist', 'cifar10']:
    args.n_classes = 10
  elif args.dataset in ['cifar100']:
    args.n_classes = 100

  ## == Add novel points params ==========
  start_point = 3
  if args.dataset in ['mnist', 'fmnist']:
    last_point = 33
  elif args.dataset in ['cifar10', 'cifar100']:
    last_point = 25

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
  # == Get FashionMNIST dataset =============
  if args.dataset == 'fmnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get CIFAR10 dataset ==================
  if args.dataset == 'cifar10':
    train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
    test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
  ## ========================================
  ## ========================================

  data = np.concatenate((X_train, X_test), axis=0)  #(70000, 784)
  labels = np.concatenate((y_train, y_test), axis=0)#(70000,)
  n_data = data.shape[0]

  # == Select seen & unseen classes ==========
  if args.seed == 1:
    seen_class = np.array([0, 1, 2, 3, 4]) 
    unseen_class = [5, 6, 7, 8, 9]
  else:
    seen_class = np.random.choice(args.n_classes, args.seen_class_num, replace=False)
    unseen_class = [x for x in list(set(labels)) if x not in seen_class]

  print('seen: {}'.format(seen_class))
  print('unseen: {}'.format(unseen_class))

  # == split data by classes =================
  class_data = {}
  for class_label in set(labels):
    class_data[class_label] = []  
  for idx, sample in enumerate(data):
    class_data[labels[idx]].append(sample)
  
  for label in class_data.keys():
    class_data[label] = np.array(class_data[label])

  for label, data in class_data.items():
    print('Label: {} -> {}'.format(label, data.shape))  


  # == Preparing train dataset and test seen data ===
  train_data = []
  test_data_seen = []
  for seen_class_item in seen_class:
    train_idx = np.random.choice(class_data[seen_class_item].shape[0], args.spc, replace=False)
    seen_data = class_data[seen_class_item][train_idx]
    class_data[seen_class_item] = np.delete(class_data[seen_class_item], train_idx, axis=0)

    train_data_class = np.concatenate((seen_data, np.full((args.spc , 1), seen_class_item)), axis=1)
    train_data.extend(train_data_class)

  train_data = np.array(train_data) #(6000, 785)
  
  np.random.shuffle(train_data)
  print('train data: {}'.format(train_data.shape))
  pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
    header=None,
    index=None
  )
  print('train data saved in {}'.format(os.path.join(args.saved, args.train_file)))

 
  all_class_to_select = seen_class.tolist()
  chunk_size = 1000
  n_chunk = int(n_data / chunk_size) 
  n_chunk_stream = n_chunk - 6
  chunks = []
  add_new_class_points = np.random.choice(np.arange(start_point, n_chunk_stream-last_point), len(unseen_class), replace=False)
  print('Novel class points: {}'.format(add_new_class_points))
  for i_chunk in range(n_chunk_stream):
    chunk_data = []
    
    # add novel class to test data pool
    if i_chunk in add_new_class_points:  
      rnd_uns_class = unseen_class[0]
      unseen_class.remove(rnd_uns_class)
      all_class_to_select.append(rnd_uns_class)
    
    # Select data from every known class
    if len(all_class_to_select) > 5:
      select_class_idx = np.random.choice(len(all_class_to_select), len(all_class_to_select)-1, replace=False)
      class_to_select = [all_class_to_select[i] for i in select_class_idx]
    else:
      class_to_select = all_class_to_select

    items_per_class = int(chunk_size / len(class_to_select))
    removed_class = []

    for known_class in class_to_select:
      n = class_data[known_class].shape[0]
      if n > items_per_class:
        idxs = np.random.choice(range(n), size=items_per_class, replace=False)  
        selected_data_class = np.concatenate((class_data[known_class][idxs], np.full((items_per_class , 1), known_class)), axis=1)
        chunk_data.extend(selected_data_class)  
        class_data[known_class] = np.delete(class_data[known_class], idxs, axis=0)
      
      else:
        selected_data_class = np.concatenate((class_data[known_class], np.full((class_data[known_class].shape[0] , 1), known_class)), axis=1)
        chunk_data.extend(selected_data_class)
        removed_class.append(known_class)
        del class_data[known_class]

    if len(removed_class) > 0:
      all_class_to_select = [e for e in all_class_to_select if e not in removed_class]

    chunk_data = np.array(chunk_data)

    # check if chunk_data < chunk_size
    if chunk_data.shape[0] < chunk_size:
      needed_data = chunk_size - chunk_data.shape[0]
      helper_class = all_class_to_select[-1]

      n = class_data[helper_class].shape[0]
      idxs = np.random.choice(range(n), size=needed_data, replace=False)
      selected_data_class = np.concatenate((class_data[helper_class][idxs], np.full((needed_data , 1), helper_class)), axis=1)
      chunk_data = np.concatenate((chunk_data, selected_data_class), axis=0)

    np.random.shuffle(chunk_data)
    chunks.append(chunk_data)
    
  stream_data = np.concatenate(chunks, axis=0)
  print('stream_data size: {}'.format(stream_data.shape))
  pd.DataFrame(stream_data).to_csv(os.path.join(args.saved, args.stream_file),
    header=None,
    index=None
  )
  print('stream data saved in {}'.format(os.path.join(args.saved, args.stream_file)))
  


























###### Old Version 
# import pandas as pd 
# import numpy as np
# import argparse
# import random
# import gzip
# import os

# def load_mnist(path, kind='train'):
#   """Load MNIST data from `path`"""
#   labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
#   images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

#   with gzip.open(labels_path, 'rb') as lbpath:
#       labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
#   with gzip.open(images_path, 'rb') as imgpath:
#       images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

#   return images, labels


# ## == Params =====================
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10]
# parser.add_argument('--class_num', type=int, default=10, help='')
# parser.add_argument('--seen_class_num', type=int, default=5, help='')
# parser.add_argument('--spc', type=int, default=1200, help='samples per class')
# parser.add_argument('--saved', type=str, default='./data/', help='')
# parser.add_argument('--seed', type=int, default=2, help='')
# args = parser.parse_args()

# # parser.add_argument('--stream_file', type=str, default='m_stream.csv', help='')
# # parser.add_argument('--data_path', type=str, default='./data/', help='')

# ## == Apply seed =================
# np.random.seed(args.seed)

# # = Add some variables to args =========
# args.data_path = 'data/{}'.format(args.dataset)
# args.train_file = '{}_train.csv'.format(args.dataset)
# args.stream_file = '{}_stream.csv'.format(args.dataset)



# if __name__ == '__main__':
#   ## ========================================
#   # == Get MNIST dataset ====================
#   if args.dataset == 'mnist':
#     train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
#     test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values

#     X_train, y_train = train_data[:, 1:], train_data[:, 0]
#     X_test, y_test = test_data[:, 1:], test_data[:, 0]
#   ## ========================================
#   ## ========================================

#   ## ========================================
#   # == Get Fashion-MNIST dataset ============
#   if args.dataset == 'fmnist':
#     train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
#     test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
#     X_train, y_train = train_data[:, 1:], train_data[:, 0]
#     X_test, y_test = test_data[:, 1:], test_data[:, 0]
#     # X_train, y_train = load_mnist(path, kind='train') #(60000, 784), (60000,)
#     # X_test, y_test = load_mnist(path, kind='t10k')    #(10000, 784), (10000,)
#   ## ========================================
#   ## ========================================

#   ## ========================================
#   # == Get Cifar10 dataset ==================
#   if args.dataset == 'cifar10':
#     train_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_train.csv'), sep=',', header=None).values
#     test_data = pd.read_csv(os.path.join(args.data_path, 'cifar10_test.csv'), sep=',', header=None).values
#     X_train, y_train = train_data[:, :-1], train_data[:, -1]
#     X_test, y_test = test_data[:, :-1], test_data[:, -1]

#   ## ========================================
#   ## ========================================

#   ## ========================================
#   # == For batch training ===================
#   # train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)   #(60000, 785)
#   # test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)   #(10000, 785)

#   # pd.DataFrame(train_data).to_csv(os.path.join(data_path,'fm_train_batch.csv'),
#   #   header=None,
#   #   index=None
#   # )
#   # pd.DataFrame(test_data).to_csv(os.path.join(data_path,'fm_test_batch.csv'),
#   #   header=None,
#   #   index=None
#   # )
#   # print('done')
#   # time.sleep(20)
#   ## ========================================
#   ## ========================================


#   data = np.concatenate((X_train, X_test), axis=0)  #(70000, 784)
#   labels = np.concatenate((y_train, y_test), axis=0)#(70000,)

#   # == split data by classes =================
#   class_data = {}
#   for class_label in set(labels):
#     class_data[class_label] = []  
#   for idx, sample in enumerate(data):
#     class_data[labels[idx]].append(sample)

#   # == Select seen & unseen classes ==========
#   seen_class = np.random.choice(args.class_num, args.seen_class_num, replace=False)
#   unseen_class = [x for x in list(set(labels)) if x not in seen_class]
#   # seen_class = [0, 1, 2, 3, 4] 
#   # unseen_class = [5, 6, 7, 8, 9]
#   print('seen: {}'.format(seen_class))
#   print('unseen: {}'.format(unseen_class))

#   # == Preparing train dataset and test seen data ===
#   train_data = []
#   test_data_seen = []
#   for seen_class_item in seen_class:
#     seen_data = np.array(class_data[seen_class_item])

#     del class_data[seen_class_item]

#     np.random.shuffle(seen_data)
#     last_idx = args.spc
#     test_data_length = seen_data.shape[0] - last_idx
#     train_part = seen_data[:last_idx]
#     test_part = seen_data[last_idx:]

#     train_data_class = np.concatenate((train_part, np.full((last_idx , 1), seen_class_item)), axis=1)
#     train_data.extend(train_data_class)

#     test_data_class = np.concatenate((test_part, np.full((test_data_length , 1), seen_class_item)), axis=1)
#     test_data_seen.extend(test_data_class)
  
#   train_data = np.array(train_data) #(6000, 785)
#   np.random.shuffle(train_data)  
#   pd.DataFrame(train_data).to_csv(os.path.join(args.saved, args.train_file),
#     header=None,
#     index=None
#   )

#   # == Preparing test(stream) dataset ================
#   test_data_seen = np.array(test_data_seen) #(30000, 785)
#   all_temp_data = []
#   add_class_point = 6000
  
#   np.random.shuffle(test_data_seen)
#   all_temp_data = test_data_seen[:add_class_point]
#   test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

#   while True:

#     if len(unseen_class) != 0:
#       rnd_uns_class = unseen_class[0]
#       unseen_class.remove(rnd_uns_class)
      
#       selected_data = np.array(class_data[rnd_uns_class])
#       del class_data[rnd_uns_class]
#       temp_data_with_label = np.concatenate((selected_data, np.full((selected_data.shape[0] , 1), rnd_uns_class)), axis=1)
#       test_data_seen = np.concatenate((test_data_seen, temp_data_with_label), axis=0)

#     np.random.shuffle(test_data_seen)
#     all_temp_data = np.concatenate((all_temp_data, test_data_seen[:add_class_point]), axis=0)
#     # all_temp_data.append(test_data_seen[:add_class_point])
#     test_data_seen = np.delete(test_data_seen, slice(add_class_point), 0)

#     if len(unseen_class) == 0:
#       np.random.shuffle(test_data_seen)
#       all_temp_data = np.concatenate((all_temp_data, test_data_seen), axis=0)
#       break
  
#   test_data = np.stack(all_temp_data)
#   pd.DataFrame(test_data).to_csv(os.path.join(args.saved, args.stream_file),
#     header=None,
#     index=None
#   )

#   # dataset = np.concatenate((train_data, test_data), axis=0)
#   # pd.DataFrame(dataset).to_csv(
#   #   os.path.join(args.saved, args.stream_file),
#   #   header=None,
#   #   index=None
#   # )