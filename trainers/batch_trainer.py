import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import os

def train(
  feature_ext,
  relation_net,
  train_loader,
  val_loader,
  args, device):

  print('training ...')
  criterion = torch.nn.CrossEntropyLoss()
  # optim = SGD(feature_ext.parameters(), lr=args.lr, momentum=args.momentum)
  optim = Adam(feature_ext.parameters(), lr=args.lr)
  scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)

  min_loss = float('inf')
  for epoch_item in range(args.start_epoch, args.epochs):
    print('=== Epoch %d ===' % epoch_item)
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
      images, labels = batch
      images, labels = images.to(device), labels.to(device)
      
      optim.zero_grad()
      outputs, _ = feature_ext.forward(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optim.step()

      train_loss += loss

      # == Validation =================
      if (i+1) % args.log_interval == 0:
        with torch.no_grad():
          total_val_loss = 0.0
          feature_ext.eval()
          for j, data in enumerate(val_loader):
            sample, labels = data
            sample, labels = sample.to(device), labels.to(device)

            logits, _ = feature_ext.forward(sample)
            loss = criterion(logits, labels)
            loss = loss.mean()
            total_val_loss += loss.item()

          total_val_loss /= len(val_loader)
          print('=== Epoch: %d/%d, Train Loss: %f, Val Loss: %f' % (
            epoch_item, i+1,  train_loss/args.log_interval, total_val_loss))
          train_loss = 0.

          # save best model
          if total_val_loss < min_loss:
            torch.save(feature_ext.state_dict(), os.path.join(args.save, "feature_ext_best.pt"))
            # torch.save(relation_net.state_dict(), os.path.join(args.save, "relation_net_best.pt"))
            min_loss = total_val_loss
            print("Saving new best model")
    
    if args.scheduler:
      scheduler.step()
    
  # save last model
  torch.save(feature_ext.state_dict(), os.path.join(args.save, "feature_ext_last.pt"))
  # torch.save(relation_net.state_dict(), os.path.join(args.save, "relation_net_last.pt"))
  print("Saving new last model")


def test(model, test_loader, args, device):
  print('Testing model ...')
  model.to(device)

  correct = 0
  total = 0

  model.eval()
  with torch.no_grad():
    for i, data in enumerate(test_loader):
  
      samples, labels = data
      samples, labels = samples.to(device), labels.to(device)
      logits, _ = model.forward(samples)
      
      _, predicted = torch.max(logits, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %7.4f %%' % (100 * correct / total))
  return correct / total

