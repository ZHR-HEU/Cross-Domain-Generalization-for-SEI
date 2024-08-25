#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from scipy import signal
# from scipy.fftpack import fft,ifft
# from scipy.fft import fftshift
# import random
import h5py
#原始文件中是在datasets.py中读取数据的，分别读取测试集和训练集
import random

def load_h5(h5_path):
    # load training data  读取数据并处理
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            # X1 = X
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")

        index = [i for i in range(len(Y1))]
        # random.seed(10)
        random.shuffle(index)
        data = X1[index, :, :]
        # data = np.expand_dims(data, axis=1)
        label = Y1[index]
        print("打乱数据集",data.shape)
        print("打乱数据集",label.shape)
        # print("打乱数据集",label[:6])
    return data, label

def load_h6(h5_path):
    # load training data  读取数据并处理
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")
        m = 800
        # list = range(1000)
        import random
        # print()  random.sample(list,m)
        Empty = np.zeros([ m*100 ,2 , 2196], dtype=np.float32)
        Empty_y = np.zeros([m*100, 1], dtype=np.float32)
        for n in range(100):
            # X = X1[1000*n+random.sample(list,m),:,:]
            # for j in range(m):
            # Empty[m*n:m-1+m*n,:,:] = X1.take(1000*n+random.sample(list,m))
            # Empty_y[m*n:m-1+m*n] = Y1.take(1000*n+random.sample(list,m))
            Empty[m*n:m-1+m*n,:,:] = X1[1000*n:m-1+1000*n,:,:]
            Empty_y[m*n:m-1+m*n] = Y1[1000*n:m-1+1000*n]
        index = [i for i in range(len(Empty_y))]
        random.shuffle(index)
        data = Empty[index,:,:]
        label = Empty_y[index]
        print("打乱数据集",data.shape)
        print("打乱数据集",label.shape)
    return data, label

class SignalDataset(Dataset):
    """数据加载器  返回指定数目的数据"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.X, self.Y = load_h6(data_folder)   # (3392, 8192, 1)

    def __getitem__(self, item):
        # 返回一个音频数据
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.X)


def load_h5_2D(h5_path):
    # load training data  读取数据并处理
    with h5py.File(h5_path, 'r') as hf:
        head = list(hf.keys())
        print('List of arrays in input file:', hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2) # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")

        index = [i for i in range(len(Y1))]
        # random.seed(10)
        random.shuffle(index)
        data = X1[index,:,:,:]
        label = Y1[index]
        print("打乱数据集",data.shape)
        print("打乱数据集",label.shape)
        # print("打乱数据集",label[:6])
    return X1, Y1

class SignalDataset1(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.X, self.Y = load_h5(data_folder)   # (3392, 8192, 1)

    def __getitem__(self, item):
        # 返回一个音频数据
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.X)

class SignalDataset2D(Dataset):
    """数据加载器"""
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.X, self.Y = load_h5_2D(data_folder)   # (3392, 8192, 1)
    def __getitem__(self, item):
        X = self.X[item]
        Y = self.Y[item]
        return X, Y
    def __len__(self):
        return len(self.X)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X, Y = load_h6('Case0_7.68M/Case0_7.68M_TX_100X1000_15dB_2196_withoutCFO.mat')
    print(Y[90:110])
    # X, Y = load_h5('Case0_7.68M_LOS_20dB.mat')  #(3845, 2, 4800)
    # # 信号长度变化
    # # len1 = np.random.randint(2400,4800)
    # # data1 = np.zeros([X.shape[0], X.shape[1], X.shape[2]])
    # #
    # # x_resampled = signal.resample(X[1,1,:], len1) # 函数使用FFT将信号重采样成n个点
    # # print(x_resampled.shape)
    # # 求取FFT
    # print(X.shape)
    # fig = plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(X[2,0,:],label="I")
    # plt.legend()
    # plt.subplot(4, 1, 2)
    # plt.plot(X[2,1,:],label="Q")
    # plt.legend()
    # plt.subplot(4, 1, 3)
    # plt.plot(X[2,2,:],label="FFT_I_abs")
    # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(X[2,3,:],label="FFT_I_angle")
    # plt.legend()
    #
    # fig = plt.figure()
    # plt.subplot(4,1,1)
    # plt.plot(X[999,0,:],label="I")
    # plt.legend()
    # plt.subplot(4, 1, 2)
    # plt.plot(X[999,1,:],label="Q")
    # plt.legend()
    # plt.subplot(4, 1, 3)
    # plt.plot(X[999,2,:],label="FFT_I_abs")
    # plt.legend()
    # plt.subplot(4, 1, 4)
    # plt.plot(X[999,3,:],label="FFT_I_angle")
    # plt.legend()
    #
    # plt.show()
