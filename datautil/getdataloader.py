# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from datautil.mydata_read import SignalDataset1
from torch.utils.data import DataLoader
import torch

def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []   #存储训练集和测试集

    names = args.img_dataset[args.dataset]  #获取不同数据集的名称
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]

    return train_loaders, eval_loaders


def get_img_dataloader1(args):
    trdatalist, tedatalist = [], []
    source1 = SignalDataset1('SUI-1.mat')
    source1_train_size = int(0.8 * len(source1))
    source1_eval_size = len(source1) - int(0.8 * len(source1))
    source1_train_data, source1_eval_data = torch.utils.data.random_split(source1,
                                                                          [source1_train_size, source1_eval_size])

    source2 = SignalDataset1('SUI-3.mat')
    source2_train_size = int(0.8 * len(source2))
    source2_eval_size = len(source2) - int(0.8 * len(source2))
    source2_train_data, source2_eval_data = torch.utils.data.random_split(source2,
                                                                          [source2_train_size, source2_eval_size])

    target = SignalDataset1('SUI-5.mat')

    trdatalist = [source1_train_data, source2_train_data]
    tedatalist = [source1_eval_data, source2_eval_data, target]
    # trdatalist = [source1_train_data]
    # tedatalist = [source1_eval_data, target]

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, eval_loaders