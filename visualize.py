import torch
from torch.utils.data import DataLoader
import os
from pandas import read_csv
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from samplers.pt_sampler import PtSampler

def set_novel_label(known_labels, args):
  
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',', header=None).values

  for idx, data in enumerate(stream_data):
    label = data[-1]
    if label not in known_labels:
      stream_data[idx, -1] = 100

  return stream_data

def tsne_plot(features, labels, file_name='tsne'):
  print('t-SNE plotting ...')
  tsne = TSNE()
  X_embedded = tsne.fit_transform(features)

  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 6)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=labels,
    legend='full',
    palette=palette
  )

  plt.savefig('{}.png'.format(file_name))
  plt.show()
  print('Done!')

def pca_plot(features, labels, file_name='pca'):
  print('PCA plotting ...')
  pca = PCA(n_components=2)
  X_embedded = pca.fit_transform(features)

  sns.set(rc={'figure.figsize':(11.7,8.27)})
  palette = sns.color_palette("bright", 10)
  sns.scatterplot(
    x=X_embedded[:,0],
    y=X_embedded[:,1],
    hue=labels,
    legend='full',
    palette=palette
  )

  plt.savefig('{}.png'.format(file_name))
  plt.show()
  print('Done!')

def visualization(model, dataset, device):
  
  # activation = {}
  # def get_activation(name):
  #   def hook(model, input, output):
  #     activation[name] = output.detach()
  #   return hook
  
  print(dataset.label_set)
  print(len(dataset))
  sampler = PtSampler(
    dataset,
    n_way=6,
    n_shot=500,
    n_query=0,
    n_tasks=1
  )
  dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )

  with torch.no_grad():
    batch = next(iter(dataloader))
    support_images, support_labels, _, _ = batch
    support_images = support_images.reshape(-1, *support_images.shape[2:])
    support_labels = support_labels.flatten()
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)

    # model.layer4[2].bn3.register_forward_hook(get_activation('features'))
    # features = torch.squeeze(activation['features'])
    
    outputs, features = model.forward(support_images)
    features = features.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()

    # for feature in features:
    #   print(feature)
    # print(support_labels)
    # print(features.shape)
    # print(support_labels.shape)
  # features += 1e-12

  tsne_plot(features, support_labels, file_name='tsne_last')
  # pca_plot(features, support_labels, file_name='pca_last')

