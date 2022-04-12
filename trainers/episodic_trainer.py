from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import time

from dataset import SimpleDataset
from utils.preparation import dataloader_preparation, transforms_preparation


def train(
  feature_ext,
  relation_net,
  learner,
  train_data,
  args, device):

  ## == train_loader For calculate PTs ====
  if args.use_transform:
    train_transform, _ = transforms_preparation()
    train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
  else:
    train_dataset = SimpleDataset(train_data, args)
  train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)

  ## == Set in learner ====================
  learner.set_items_per_label(train_dataset)

  ## == Loaders for training ==============
  train_dataloader,\
    val_dataloader,\
      known_labels = dataloader_preparation(train_data, [], args)
  
  feature_ext_optim = Adam(feature_ext.parameters(), lr=args.lr_ext)
  feature_ext_scheduler = StepLR(feature_ext_optim, step_size=args.step_size, gamma=args.gamma)
  relation_net_optim = Adam(relation_net.parameters(), lr=args.lr_rel)
  relation_net_scheduler = StepLR(relation_net_optim, step_size=args.step_size, gamma=args.gamma)

  global_time = time.time()
  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)
      train_ext_loss = 0.
      train_rel_loss = 0.
      
      trainloader = iter(train_dataloader)

      for miteration_item in range(args.meta_iteration):
        batch = next(trainloader)
        
        ext_loss = learner.feature_ext_train(
          feature_ext,
          batch,
          feature_ext_optim,
          args)
        train_ext_loss += ext_loss

        ## == train relation ==========
        # if (miteration_item + 1) % args.relation_train_interval == 0:
        #   # learner.calculate_prototypes(feature_ext, train_loader)
        #   learner.calculate_examplers(feature_ext, train_dataset)

        #   rel_loss = learner.relation_train(
        #     feature_ext,
        #     relation_net,
        #     batch,
        #     feature_ext_optim,
        #     relation_net_optim,
        #     known_labels,
        #     args)
        #   train_rel_loss += rel_loss
          
        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          train_loss_total = (train_ext_loss+train_rel_loss) / args.log_interval
          train_ext_loss = 0.
          train_rel_loss = 0.

          if args.rep_approach == 'prototype':
            learner.calculate_prototypes(feature_ext, train_loader)
          elif args.rep_approach == 'exampler':
            learner.calculate_examplers(feature_ext, train_dataset, k=args.n_examplers)

          # evalute on val_dataset
          val_loss, \
          val_cw_acc, \
          val_ow_acc \
            = learner.evaluate(
              feature_ext,
              relation_net,
              val_dataloader,
              known_labels,
              args)

          # print losses
          # print('scheduler: %f' % (optim.param_groups[0]['lr']))
          print('=== Time: %.2f, Step: %d, TrainLoss: %f, ValLoss: %f, Val-CwAcc: %.2f, Val-OwAcc: %.2f' % (
            time.time()-global_time,
            miteration_item+1,
            train_loss_total,
            val_loss,
            val_cw_acc*100,
            val_ow_acc*100
          ))
          global_time = time.time()
    
          # save best feature_ext
          if val_loss < min_loss:
            feature_ext.save(os.path.join(args.save, "feature_ext_best.pt"))
            relation_net.save(os.path.join(args.save, "relation_net_best.pt"))
            min_loss = val_loss
            print("= ...New best model saved")
          
        if args.scheduler:
          feature_ext_scheduler.step()
          relation_net_scheduler.step()
  
  except KeyboardInterrupt:
    print('skipping training')
  
  if args.rep_approach == 'prototype':
    learner.calculate_prototypes(feature_ext, train_loader)
  elif args.rep_approach == 'exampler':
    learner.calculate_examplers(feature_ext, train_dataset, k=args.n_examplers)

  # save last model
  feature_ext.save(os.path.join(args.save, "feature_ext_last.pt"))
  relation_net.save(os.path.join(args.save, "relation_net_last.pt"))
  print("= ...last model saved")

