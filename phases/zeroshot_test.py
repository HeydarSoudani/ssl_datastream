def zeroshot_test(feature_ext,
                  relation_net,
                  detector,
                  args,
                  device,
                  known_labels=None):
  print('================================ Zero-Shot Test ================================')
  feature_ext.eval()
  relation_net.eval()
  
  # == Load stream data ==============================
  stream_data = read_csv(
    os.path.join(args.data_path, args.stream_file),
    sep=',',
    header=None).values
  if args.use_transform:
    _, test_transform = transforms_preparation()
    stream_dataset = SimpleDataset(stream_data, args, transforms=test_transform)
  else:
    stream_dataset = SimpleDataset(stream_data, args)
  dataloader = DataLoader(dataset=stream_dataset, batch_size=1, shuffle=False)

