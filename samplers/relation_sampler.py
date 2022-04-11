import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset


class RelationSampler(Sampler):
  def __init__(
    self,
    dataset: Dataset,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_tasks: int
  ):
    super().__init__(data_source=None)
    self.dataset = dataset
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.n_tasks = n_tasks

    self.items_per_label = {}
    assert hasattr(
      dataset, "labels"
    ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
    for item, label in enumerate(dataset.labels):
      if label in self.items_per_label.keys():
        self.items_per_label[label].append(item)
      else:
        self.items_per_label[label] = [item]
    
  def __len__(self):
    return self.n_tasks
  
  def __iter__(self):
    for _ in range(self.n_tasks): 
      support_samples = [  
        torch.tensor(
          random.sample(
            self.items_per_label[label], self.n_shot
          )
        )
        for label in random.sample(self.items_per_label.keys(), self.n_way)
      ]
      query_samples = [
        torch.tensor(
          random.sample(
            list(self.dataset.labels), self.n_query
          )
        )
      ]
      yield torch.cat((support_samples, query_samples))
      
  def episodic_collate_fn(self, input_data):
    all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
    all_labels = torch.tensor([x[1] for x in input_data])
    print('all_labels')
    print(all_labels)

    query_images, support_images = torch.split(
      all_images,
      [ self.n_query, self.n_way * self.n_shot]
    )
    query_labels, support_labels = torch.split(
      all_labels,
      [self.n_query, self.n_way * self.n_shot]
    )
    
    return (
      support_images,
      support_labels,
      query_images,
      query_labels
    )