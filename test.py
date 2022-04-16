import torch

def cos_similarity(x, y):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception
  
  sim = torch.zeros((n, m))

  for i in range(n):
    for j in range(m):
      sim[i, j] = torch.dot(x[i], y[j])/(torch.norm(x[i])*torch.norm(y[j]))

  return sim


if __name__ == '__main__':
    a = torch.rand(10, 128)
    b = torch.rand(5, 128)
    sim = cos_similarity(a, b)
    
    print(sim)
    print(sim.shape)