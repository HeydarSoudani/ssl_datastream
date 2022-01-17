import random
from typing import List, Tuple

import torch
from torch.utils.data import Sampler, Dataset


def LD2DT(LD):
  return {k: torch.stack([dic[k] for dic in LD]) for k in LD[0]}


class ReptileSampler(Sampler):
  """
  Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
  n_way classes, and then sample support data from these classes.
  """

  def __init__(
    self,
    dataset: Dataset,
    n_way: int,
    n_shot: int,
    n_tasks: int,
    reptile_step: int = 3,
  ):
    """
    Args:
      dataset: dataset from which to sample classification tasks. Must have a field 'label': a
        list of length len(dataset) containing containing the labels of all images.
      n_way: number of classes in one task
      n_shot: number of support data for each class in one task
      n_tasks: number of tasks to sample
    """
    super().__init__(data_source=None)
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_tasks = n_tasks
    self.reptile_step = reptile_step
    self.replacement = False

    self.indices_per_label = {}
    assert hasattr(
      dataset, "labels"
    ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
    for item, label in enumerate(dataset.labels):
      if label in self.indices_per_label.keys():
        self.indices_per_label[label].append(item)
      else:
        self.indices_per_label[label] = [item]

  def __len__(self):
    return self.n_tasks

  def __iter__(self):
    for _ in range(self.n_tasks):
      yield torch.cat(
        [
          torch.tensor(
            random.sample(
              self.indices_per_label[label],
              self.reptile_step * self.n_shot,
            )
          )
          for label in random.sample(self.indices_per_label.keys(), self.n_way)
        ]
      )

  def episodic_collate_fn(
    self, input_data: List[Tuple[torch.Tensor, int]]
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to be used as argument for the collate_fn parameter of episodic data loaders.
    Args:
      input_data: each element is a tuple containing:
        - an image as a torch Tensor
        - the label of this image
    Returns:
      list({key: Tensor for key in input_data}) with length of reptile_step
    """
    
    input_data.sort(key = lambda input_data: input_data[1])
    
    all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
    all_images = all_images.reshape(
      (self.n_way, self.n_shot*self.reptile_step, *all_images.shape[1:])
    )
    all_labels = torch.tensor(
      [x[1] for x in input_data]
    ).reshape((self.n_way, self.n_shot*self.reptile_step))
    
    
    all_images_splitted = torch.split(all_images, self.n_shot, dim=1)
    all_labels_splitted = torch.split(all_labels, self.n_shot, dim=1)

    data = [
      {'data': all_images_splitted[i], 'label': all_labels_splitted[i]}
      for i in range(self.reptile_step)
    ]

    return data