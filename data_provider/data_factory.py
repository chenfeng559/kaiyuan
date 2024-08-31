import os
<<<<<<< HEAD

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
=======
import mindspore.dataset as ds
from mindspore import log as logger
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, \
    Dataset_Custom, Dataset_PEMS, UCRAnomalyloader
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

<<<<<<< HEAD
    if args.task_name == 'forecast' or args.task_name  == 'predict':
        if args.use_ims:
            print('*******************CIAutoRegressionDatasetBenchmark')
=======
    if args.task_name in ['forecast', 'predict']:
        if args.use_ims:
            logger.info('*******************CIAutoRegressionDatasetBenchmark')
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
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
<<<<<<< HEAD
            print('*******************CIDatasetBenchmark')
=======
            logger.info('*******************CIDatasetBenchmark')
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
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
<<<<<<< HEAD
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
=======

        logger.info(f"{flag} {len(data_set)}")

        # 使用 MindSpore 的数据集
        dataset = ds.GeneratorDataset(data_set, column_names=["data", "label"], shuffle=shuffle_flag)
        dataset = dataset.batch(batch_size, drop_remainder=drop_last)

        return data_set, dataset
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

    elif args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = UCRAnomalyloader(
            root_path=args.root_path,
            data_path=args.data_path,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            flag=flag,
        )
<<<<<<< HEAD
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
=======
        logger.info(f"{flag} {len(data_set)}")
        dataset = ds.GeneratorDataset(data_set, column_names=["data", "label"], shuffle=shuffle_flag)
        dataset = dataset.batch(batch_size, drop_remainder=drop_last)
        return data_set, dataset

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
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
<<<<<<< HEAD
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
=======
        logger.info(f"{flag} {len(data_set)}")
        dataset = ds.GeneratorDataset(data_set, column_names=["data", "label"], shuffle=shuffle_flag)
        dataset = dataset.batch(batch_size, drop_remainder=drop_last)
        return data_set, dataset

>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    else:
        raise NotImplementedError
