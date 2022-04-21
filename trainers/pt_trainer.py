from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import time

from dataset import SimpleDataset
from utils.preparation import dataloader_preparation, transforms_preparation

def pt_training(
  feature_ext,
  learner,
  optim,
  train_dataloader,
  val_dataloader,
  known_labels,
  scheduler,
  args
):
  global_time = time.time()
  min_loss = float('inf')

  train_ext_loss = 0.

  print('===== Feature extractor fine-tuning ... =====')
  trainloader = iter(train_dataloader)
  for miteration_item in range(args.meta_iteration):
    batch = next(trainloader)

    ext_loss = learner.train(
      feature_ext,
      batch,
      optim,
      args)
    train_ext_loss += ext_loss

    # = validation ===
    if (miteration_item + 1) % args.log_interval == 0:
      train_loss_total = train_ext_loss / args.log_interval
      train_ext_loss = 0.

      # evalute on val_dataset
      val_loss, \
      val_cw_acc, \
      val_ow_acc \
        = learner.evaluate(
          feature_ext,
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
        min_loss = val_loss
        print("= ...New best model saved")
  
    if args.scheduler:
      scheduler.step()


def train(
  feature_ext,
  learner,
  train_data,
  args):

  ## == train_loader For calculate PTs ====
  if args.use_transform:
    train_transform, _ = transforms_preparation()
    train_dataset = SimpleDataset(train_data, args, transforms=train_transform)
  else:
    train_dataset = SimpleDataset(train_data, args)
  train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)

  ## == Loaders for training ==============
  train_dataloader,\
    val_dataloader,\
      known_labels = dataloader_preparation(train_data, [], args)

  optim = SGD(feature_ext.parameters(), lr=args.lr, momentum=args.momentum)
  scheduler = StepLR(optim, step_size=args.step_size, gamma=args.gamma)

  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print('=== Epoch %d ===' % epoch_item)

      pt_training(
        feature_ext,
        learner,
        optim,
        train_dataloader,
        val_dataloader,
        known_labels,
        scheduler,
        args
      )
  except KeyboardInterrupt:
    print('skipping training')


  # == Save last model ========
  feature_ext.save(os.path.join(args.save, "feature_ext_last.pt"))
  print("= ...last model saved")

  # == Save learner ===========
  learner.calculate_prototypes(feature_ext, train_loader)
  learner.save(os.path.join(args.save, "learner.pt"))
  print("= ...learner saved")


