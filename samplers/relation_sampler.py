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
      classes = random.sample(self.items_per_label.keys(), self.n_way)
      query_class = random.sample(classes, 1)
      support_class = classes.remove(query_class)

      print(classes)
      print(query_class)
      print(support_class)

      query_class_samples =  torch.tensor(
        random.sample(
          self.items_per_label[query_class], self.n_shot + self.n_query
        )
      )
      
      support_class_samples = torch.cat(
        [
          torch.tensor(
            random.sample(
              self.items_per_label[label], self.n_shot
            )
          )
          for label in random.sample(self.items_per_label.keys(), support_class)
        ]
      )
      print(query_class_samples.shape)
      print(support_class_samples.shape)
      yield torch.cat(support_class_samples, query_class_samples)
  
  def episodic_collate_fn(self, input_data):
    print(input_data)
    return 0