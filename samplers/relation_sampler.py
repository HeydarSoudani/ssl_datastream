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
      # print(self.items_per_label.keys())
      # print(self.n_way)
      # print(random.sample(self.items_per_label.keys(), self.n_way))
      classes = random.sample(self.items_per_label.keys(), self.n_way)
      query_class = random.sample(classes, 1)[0]
      classes.remove(query_class)

      # print(classes)
      # print(query_class)
      # print(support_class)

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
          for label in classes
        ]
      )
      yield torch.cat((query_class_samples, support_class_samples))
  
  def episodic_collate_fn(self, input_data):
    all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
    all_labels = torch.tensor([x[1] for x in input_data])

    print(all_images.shape)
    print(all_labels)

    query_images, support_images = torch.split(all_images, self.n_query)
    print(query_images.shape)
    print(support_images.shape)

    query_labels, support_labels = torch.split(all_labels, self.n_query)
    print(query_labels)
    print(support_labels)

    return 0