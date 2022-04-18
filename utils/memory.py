import torch
import numpy as np

class OperationalMemory():
  def __init__(self,
                device,
                selection_type='fixed_mem',   # ['fixed_mem', 'pre_class']
                total_size=1000,
                per_class=100,
                novel_acceptance=150,
                selection_method='rand'):
    self.selection_type = selection_type
    self.total_size = total_size
    self.per_class = per_class
    self.novel_acceptance = novel_acceptance
    self.device = device
    self.selection_method = selection_method
    self.class_data = None

  def select(self, data, return_data=False):
    """
    Compute ...
    Args:
      data: list of (sample, label)
    Returns:
      ---
    """ 
    samples = torch.stack([item[0] for item in data])
    labels = torch.tensor([item[1] for item in data])
    seen_labels = torch.unique(labels)

    ### === Seperate new data =====================
    new_class_data = {
      l.item(): samples[(labels == l).nonzero(as_tuple=True)[0]]
      for l in seen_labels
    }

    ### == All data together ======================
    if self.class_data != None:
      # == if not first time
      # should add buffer data
      keys = set(torch.tensor(list(self.class_data.keys())).tolist() + \
      torch.tensor(list(new_class_data.keys())).tolist())
      known_keys = set(torch.tensor(list(self.class_data.keys())).tolist())
      new_keys = set(torch.tensor(list(new_class_data.keys())).tolist())

      for key in keys:
        if key in known_keys:
          if key in new_keys:
            self.class_data[key] = torch.cat((self.class_data[key], new_class_data[key]), 0)
        else:
          self.class_data[key] = new_class_data[key]
    else:
      # if first time
      self.class_data = new_class_data  

    
    ### == Random selection ========================
    if self.selection_method == 'rand':
      if self.selection_type == 'fixed_mem':
        unique_labels = list(self.class_data.keys())
        class_size = int(self.total_size / len(unique_labels))

        for label, samples in self.class_data.items():
          n = samples.shape[0]
          if n >= class_size:
            idxs = np.random.choice(range(n), size=class_size, replace=False)
            self.class_data[label] = samples[idxs]
          else:
            self.class_data[label] = samples

      elif self.selection_type == 'pre_class':
        self.rand_selection()

    # elif self.selection_method == 'soft_rand':
    #   self.soft_rand_selection() 
    # for label, features in self.class_data.items():
    #   print('{} -> {}'.format(label, features.shape))
    
    ### == Returning data in appropiate form ========
    if return_data:
      returned_data_list = []
      for label, samples in self.class_data.items():
        n = samples.shape[0]
        if n >= self.novel_acceptance:
          samples = samples.reshape(samples.shape[0], -1)*255
          labels = torch.full((n, 1), label, device=self.device, dtype=torch.float) #[200, 1]
          data = torch.cat((samples, labels), axis=1)
          returned_data_list.append(data)
      
      returned_data = torch.cat(returned_data_list, 0)
      returned_data = returned_data.detach().cpu().numpy()
      np.random.shuffle(returned_data)
      return returned_data.int()

  def rand_selection(self):
    for label, samples in self.class_data.items():
      n = samples.shape[0]
      if n >= self.per_class:
        idxs = np.random.choice(range(n), size=self.per_class, replace=False)
        self.class_data[label] = samples[idxs]

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)

