import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import os
import time


def train(
  feature_ext,
  relation,
  learner,
  train_loader,
  val_loader,
  known_labels,
  args,
  device):
  
  # criterion = torch.nn.CrossEntropyLoss()
  # optim = SGD(feature_ext.parameters(), lr=args.lr, momentum=args.momentum)
  optim = Adam(feature_ext.parameters(), lr=args.lr)
  scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)

  global_time = time.time()
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)
      train_loss = 0.
      trainloader = iter(train_loader)

      for miteration_item in range(args.meta_iteration):
        batch = next(trainloader)
        loss = learner.train(
          feature_ext,
          relation,
          batch,
          optim,
          miteration_item,
          args)
        train_loss += loss

        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          
          train_loss_total = train_loss / args.log_interval
          train_loss = 0.

          # evalute on val_dataset
          val_loss_total, \
          val_acc_dis_total, \
          val_acc_cls_total = learner.evaluate(feature_ext, val_loader, known_labels, args)  # For Pt.

          # print losses
          # print('scheduler: %f' % (optim.param_groups[0]['lr']))
          print('=== Time: %.2f, Step: %d, Train Loss: %f, Val Loss: %f' % (
            time.time()-global_time, miteration_item+1, train_loss_total, val_loss_total))
          # print('===============================================')
          global_time = time.time()
    
          # save best feature_ext
          if val_loss_total < min_loss:
            torch.save(feature_ext.state_dict(), os.path.join(args.save, "model_best.pt"))
            min_loss = val_loss_total
            print("= ...New best model saved")
          
          if args.scheduler:
            scheduler.step()
  
  except KeyboardInterrupt:
    print('skipping training')

  # save last model
  torch.save(feature_ext.state_dict(), os.path.join(args.save, "model_last.pt"))
  print("= ...New last model saved")



def test(model, train_loader, args, device):
  pass