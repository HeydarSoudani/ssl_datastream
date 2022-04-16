import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import SimpleDataset
from losses import cos_similarity



class SimDetector(object):
  def __init__(self, device):
    self.device = device

  def __call__(self, feature, representors, rep_per_class):
    detected_novelty = False
    
    all_sim = cos_similarity(feature, representors)
    # prob, predicted_idx = torch.max(all_sim, 1)
    # predicted_label = self._known_labels[torch.div(predicted_idx, rep_per_class, rounding_mode='trunc')]
    
    avg_sim = torch.tensor([
      torch.mean(all_sim[:, i*rep_per_class:(i+1)*rep_per_class])
      for i in self._known_labels
    ])
    prob, predicted_idx = torch.max(avg_sim, 0)
    predicted_label = self._known_labels[predicted_idx]

    if prob < self.thresholds[predicted_label]:
      detected_novelty = True
      predicted_label = -1
      prob = 0.0
    
    return detected_novelty, predicted_label, prob, avg_sim
    
    # reps_dict = { l: representors[l] for l in self._known_labels }

    # pts = torch.cat(list(pts_dict.values()))
    # labels = torch.tensor(list(pts_dict.keys()))
    # dists = torch.cdist(feature.reshape(1, -1), pts).flatten()
    # probs = torch.nn.functional.softmax(-dists)

    # idx = torch.argmin(dists)
    # min_dist = torch.min(dists)
    # predicted_label = labels[idx].item()
    # prob = probs[idx]

    # if min_dist > self.thresholds[predicted_label]:
    #   detected_novelty = True
    #   predicted_label = -1
    #   prob = 0.0
      
    # return detected_novelty, predicted_label, prob
  
  def set_known_labels(self, label_set):
    self._known_labels = set(label_set)
  
  def threshold_calculation(self, data, feature_ext, representors, known_labels, args):
    self._known_labels = set(known_labels)
    features_per_class = {l: [] for l in self._known_labels}
    dataset = SimpleDataset(data, args)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
      for i, batch in enumerate(dataloader):
        image, label = batch
        label = label.flatten()
        image, label = image.to(self.device), label.to(self.device)
        _, feature = feature_ext(image)
        features_per_class[label.item()].append(feature.detach())

    self.thresholds = {
      l: round(cos_similarity(torch.cat(features_per_class[l]), representors[l]).mean().item(), 4)
      for l in self._known_labels
    }

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)

