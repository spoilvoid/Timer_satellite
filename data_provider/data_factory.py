import os
import os.path as osp
import random
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, \
    Dataset_Custom, Dataset_PEMS, UCRAnomalyloader, MultivariateDatasetBenchmark, MultivariateAnomalyDatasetBenchmark
from data_provider.data_loader_benchmark import CIDatasetBenchmark, \
    CIAutoRegressionDatasetBenchmark

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'UCRA': UCRAnomalyloader,
    'multivariate': MultivariateDatasetBenchmark,
    'multivariate_anomaly': MultivariateAnomalyDatasetBenchmark,
}


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'forecast':
        if args.model == 'Timer_multivariate':
            Data = data_dict[args.data]
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
            )
        elif args.use_ims:
            data_set = CIAutoRegressionDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.output_len if flag == 'test' else args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        else:
            data_set = CIDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                pred_len=args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False)
        return data_set, data_loader

    elif args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = UCRAnomalyloader(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'imputation':
        Data = data_dict[args.data]
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError


def concat_data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1  # bsz=1 for evaluation
        root_path = osp.join(args.root_path, 'test')
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
        root_path = osp.join(args.root_path, 'trainval')
    
    if args.task_name == 'forecast':
        # dataset define
        dataset_list = []
        for data_path in os.listdir(root_path):
            dataset = MultivariateDatasetBenchmark(
                root_path=root_path,
                data_path=data_path,
                seq_len=args.seq_len,
                input_len=args.input_len,
                output_len=args.output_len,
            )
            dataset_list.append(dataset)
        combined_dataset = ConcatDataset(dataset_list)
        if flag == 'test':
            if args.use_multi_gpu:
                test_datasampler = DistributedSampler(combined_dataset, shuffle=shuffle_flag)
                test_data_loader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    sampler=test_datasampler,
                    num_workers=args.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=drop_last,
            )
            else:
                test_data_loader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last
                )
            return combined_dataset, test_data_loader
        else:
            train_size = int((1 - args.valid_ratio) * len(combined_dataset))
            val_size = len(combined_dataset) - train_size
            train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

            if args.use_multi_gpu:
                train_datasampler = DistributedSampler(train_dataset, shuffle=shuffle_flag)
                train_data_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=train_datasampler,
                    num_workers=args.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=False,
            )
            else:
                train_data_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=False
                )
            if args.use_multi_gpu:
                val_datasampler = DistributedSampler(val_dataset, shuffle=shuffle_flag)
                val_data_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    sampler=val_datasampler,
                    num_workers=args.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=False,
            )
            else:
                val_data_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=False
                )
            return train_dataset, train_data_loader, val_dataset, val_data_loader
    elif args.task_name == 'imputation':
        dataset_list = []
        for data_path in os.listdir(root_path):
            dataset = MultivariateDatasetBenchmark(
                root_path=root_path,
                data_path=data_path,
                seq_len=args.seq_len,
                input_len=args.input_len,
                output_len=args.output_len,
            )
            dataset_list.append(dataset)
        combined_dataset = ConcatDataset(dataset_list)
    
        if flag == 'test':
            test_data_loader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return combined_dataset, test_data_loader
        else:
            train_size = int((1 - args.valid_ratio) * len(combined_dataset))
            val_size = len(combined_dataset) - train_size
            train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

            train_data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False
            )
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False
            )
            return train_dataset, train_data_loader, val_dataset, val_data_loader
    elif args.task_name == 'anomaly_detection':
        if flag == 'test':
            dataset = MultivariateAnomalyDatasetBenchmark(
                root_path=root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag='test',
            )

            test_data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return dataset, test_data_loader
        else:
            train_file_num = int((1 - args.valid_ratio) * len(os.listdir(root_path)))
            train_data_path_list = random.choices(os.listdir(root_path), k=train_file_num)
            val_data_path_list = [data_path for data_path in os.listdir(root_path) if data_path not in train_data_path_list]
            
            train_dataset_list, val_dataset_list = [], []
            for data_path in train_data_path_list:
                dataset = MultivariateAnomalyDatasetBenchmark(
                    root_path=root_path,
                    data_path=data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag='train',
                )
                train_dataset_list.append(dataset)
            for data_path in val_data_path_list:
                dataset = MultivariateAnomalyDatasetBenchmark(
                    root_path=root_path,
                    data_path=data_path,
                    seq_len=args.seq_len,
                    patch_len=args.patch_len,
                    flag='val',
                )
                val_dataset_list.append(dataset)
            train_dataset = ConcatDataset(train_dataset_list)
            val_dataset = ConcatDataset(val_dataset_list)

            train_data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False
            )
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False
            )
            return train_dataset, train_data_loader, val_dataset, val_data_loader