import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
# from FC48_Adjust import AnomalyDetectionDataLoader

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode





        # data = AnomalyDetectionDataLoader(data_directory=r'D:\lzj\48FC')
        # train_loader, test_loader = data.get_dataloaders()
        # for data in train_loader:
        #     inputs, labels = data  # 假设每个批次的数据是(inputs, labels)的形式
        # for data1 in test_loader:
        #     inputs1, label1 = data1
        #
        # selftest_labels = label1
        # selftest_labels = selftest_labels.numpy()
        #
        # X_train = inputs
        # D1wei, _, D2wei = X_train.size()
        # X_train = (X_train.view(D1wei, D2wei)).numpy()
        # X_test = inputs1
        # D1weit, _, D2weit = X_test.size()
        # X_test = (X_test.view(D1weit, D2weit)).numpy()
        #
        # self.train = X_train
        # # self.train_labels = attack_labels[randIdx[:N_train]]
        #
        # self.test_labels = selftest_labels
        #
        # self.test = X_test

        # # cw_5_7.5
        # data_r10_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
        # data_r7dot5_normal = np.loadtxt('D:/数据_刘嘉帅/r=7.5 无故障/data.txt', dtype=str, delimiter=',')
        # data_r5_normal = np.loadtxt('D:/数据_刘嘉帅/r=9 无故障/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt(data_path, dtype=str, delimiter=',')
        #
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=7.5 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')
        # # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')
        #
        # # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=7.5 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')
        # # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.25t/data.txt', dtype=str, delimiter=',')
        # # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 250/data.txt', dtype=str, delimiter=',')
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 zhengtai/data.txt', dtype=str, delimiter=',')
        #
        # data_r10_normal = np.delete(data_r10_normal, np.array([0, 5, 8]), 1)
        # data_r7dot5_normal = np.delete(data_r7dot5_normal, np.array([0, 5, 8]), 1)
        # data_r5_normal = np.delete(data_r5_normal, np.array([0, 5, 8]), 1)
        # data_r10_abnormal = np.delete(data_r10_abnormal, np.array([0, 5, 8]), 1)
        # data_r7dot5_abnormal = np.delete(data_r7dot5_abnormal, np.array([0, 5, 8]), 1)
        # data_r5_abnormal = np.delete(data_r5_abnormal, np.array([0, 5, 8]), 1)
        #
        # _all_length = 160000
        # matrix_nofault = np.concatenate((data_r10_normal[:_all_length, :],
        #                                  data_r7dot5_normal[:_all_length, :],
        #                                  data_r5_normal[:_all_length, :]), axis=0)
        # matrix_addfault = np.concatenate((data_r10_abnormal[:_all_length, :],
        #                                   data_r7dot5_abnormal[:_all_length, :],
        #                                   data_r5_abnormal[:_all_length, :]), axis=0)
        #
        # # 归一化
        # def min_max_normalize_columns(data_column, min_val, max_val):
        #     data_column = data_column.astype(float)  # Convert data to float type
        #     return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
        #             max_val - min_val) + min_val
        #
        # aaaa = _all_length // 2
        # slid_window = 400
        # matrix_nofault = min_max_normalize_columns(matrix_nofault, 0, 1)
        # matrix_addfault = min_max_normalize_columns(matrix_addfault, 0, 1)
        # matrix_nofault1 = np.reshape(matrix_nofault, (3, _all_length, -1))
        # matrix_nofault2_train = matrix_nofault1[:, :_all_length // 2, :]  # (3,80000,6)
        # matrix_nofault2_test = matrix_nofault1[:, _all_length // 2:, :]
        # matrix_nofault3_train = np.reshape(matrix_nofault2_train, (240000, -1))
        # matrix_nofault3_test = np.reshape(matrix_nofault2_test, (240000, -1))
        # matrix_nofault4_train = np.reshape(matrix_nofault3_train, (slid_window, 240000 // slid_window, -1))
        # matrix_nofault4_test = np.reshape(matrix_nofault3_test, (slid_window, 240000 // slid_window, -1))
        #
        # matrix_nofault5_train = np.transpose(matrix_nofault4_train, (1, 2, 0))  # (600,6,400)
        # matrix_nofault5_test = np.transpose(matrix_nofault4_test, (1, 2, 0))  #
        #
        # matrix_addfault1 = np.reshape(matrix_addfault, (slid_window, 480000 // slid_window, -1))
        # matrix_addfault2 = np.transpose(matrix_addfault1, (1, 2, 0))
        # matrix_test = np.vstack((matrix_nofault5_test, matrix_addfault2))
        # train_labels = np.vstack((np.zeros((600, 1))))
        # test_labels = np.vstack((np.zeros((600, 1)), np.ones((1200, 1))))
        #
        # self.train = matrix_nofault5_train
        # # self.train = self.train.reshape(-1, 9, 16)
        # self.train_labels = train_labels
        #
        # self.test = matrix_test
        # self.test_labels = test_labels





        # # cw 由matlab代码转换
        # # 收集采集pos、速度和力矩数据
        # data_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
        # data_abnormal = np.loadtxt(data_path, dtype=str,  delimiter=',')  # data: (200, 1601)
        # # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')
        #
        # # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.25t\\data.txt', dtype=str, delimiter=',')
        # # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 250\\data.txt', dtype=str, delimiter=',')
        # # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 zhengtai\\data.txt', dtype=str, delimiter=',')
        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.02sin(2pi-200)\\data.txt', dtype=str, delimiter=',')
        #
        # silde_window = 400  # 400（1） #800（0）#1600（1）
        # all_length = 160000
        # data_length = np.int(all_length / silde_window)
        # data_nofault_pos = data_normal[:all_length, 2]
        # spe_x1 = data_normal[:all_length, 1]
        # liju_x1 = data_normal[:all_length, 6]
        #
        # data_GWF_pos = data_abnormal[:all_length, 2]
        # spe_x2 = data_abnormal[:all_length, 1]
        # liju_x2 = data_abnormal[:all_length, 6]
        # # 归一化*100*
        # def min_max_normalize_columns(data_column, min_val, max_val):
        #     data_column = data_column.astype(float)  # Convert data to float type
        #     return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
        #                 max_val - min_val) + min_val
        #
        # data_nofault_pos = min_max_normalize_columns(data_nofault_pos, 0, 1)
        # spe_x1 = min_max_normalize_columns(spe_x1, 0, 1)
        # liju_x1 = min_max_normalize_columns(liju_x1, 0, 1)
        # data_GWF_pos = min_max_normalize_columns(data_GWF_pos, 0, 1)
        # spe_x2 = min_max_normalize_columns(spe_x2, 0, 1)
        # liju_x2 = min_max_normalize_columns(liju_x2, 0, 1)
        # # 再合并速度、位置、力矩
        # # 将数据重新形状为100行、1600列（100，1600）
        #
        # reshaped_data_nofault_pos = np.reshape(data_nofault_pos, (data_length, silde_window))
        # reshaped_data_nofault_spe = np.reshape(spe_x1, (data_length, silde_window))
        # reshaped_data_nofault_liju = np.reshape(liju_x1, (data_length, silde_window))
        # reshaped_data_GWF_pos = np.reshape(data_GWF_pos, (data_length, silde_window))
        # reshaped_data_GWF_spe = np.reshape(spe_x2, (data_length, silde_window))
        # reshaped_data_GWF_liju = np.reshape(liju_x2, (data_length, silde_window))
        #
        # # 将三个矩阵连接起来
        # matrix_nofault = np.concatenate(
        #     (reshaped_data_nofault_pos, reshaped_data_nofault_spe, reshaped_data_nofault_liju), axis=1)
        # matrix_data_GWF = np.concatenate((reshaped_data_GWF_pos, reshaped_data_GWF_spe, reshaped_data_GWF_liju), axis=1)
        #
        # matrix_nofault= np.hstack((matrix_nofault, np.zeros((data_length,1))))
        # matrix_data_GWF = np.hstack((matrix_data_GWF, np.ones((data_length,1))))
        # # 使用 vstack 将两个矩阵垂直拼接在一起
        # merged_data = np.vstack((matrix_nofault, matrix_data_GWF))
        # data = merged_data
        #
        # # #cw
        # # data = np.loadtxt(data_path, delimiter=',')#data: (200, 1601)
        # labels = data[:, -1]  # 取最后一列数据
        # print("labels:", labels.shape)#labels: (200,)
        # features = data[:,:-1]#取除最后一列外的所有列 #(200, 1600)
        #
        # N, D = features.shape
        #
        # normal_data = features[labels==1]
        # normal_labels = labels[labels==1]
        #
        # N_normal = normal_data.shape[0]
        #
        # attack_data = features[labels==0]
        # attack_labels = labels[labels==0]
        #
        # N_attack = attack_data.shape[0]
        #
        # randIdx = np.arange(N_attack)
        # print("randIdx", randIdx,randIdx.shape)
        # np.random.shuffle(randIdx)
        # N_train = N_attack // 2
        #
        # self.train = attack_data[randIdx[:N_train]]
        # self.train = self.train.reshape(-1,3,silde_window)
        # self.train_labels = attack_labels[randIdx[:N_train]]
        #
        # self.test = attack_data[randIdx[N_train:]]
        # self.test_labels = attack_labels[randIdx[N_train:]]
        #
        # self.test = np.concatenate((self.test, normal_data),axis=0)
        # self.test = self.test.reshape(-1, 3, silde_window)
        #
        # # original_array = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
        # # # 使用reshape函数将其转换为（50，3，1600）的形状
        # # new_array = original_array.reshape((3, 2, 3))
        # # print("new_array",new_array)
        #
        # self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)
        # # self.test = np.concatenate((self.test, normal_data[0:50]), axis=0)
        # # self.test_labels = np.concatenate((self.test_labels, normal_labels[0:50]), axis=0)
        # print("self.test_labels",self.test_labels)
        # print(self.train.shape)






        # cw_5_7.5
        # data_r10_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
        # data_r7dot5_normal = np.loadtxt('D:/数据_刘嘉帅/r=7.5 无故障/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_normal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 无故障/data.txt', dtype=str, delimiter=',')
        # data_r5_normal = np.loadtxt('D:/数据_刘嘉帅/r=9 无故障/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt('./data.txt', dtype=str, delimiter=',')

        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 0.25t/data.txt', dtype=str, delimiter=',')  # DDA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 250/data.txt', dtype=str, delimiter=',')#FBA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 zhengtai/data.txt', dtype=str, delimiter=',')  # ADA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 25/data.txt', dtype=str, delimiter=',')#GWA

        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 0.25t/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 250/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 zhengtai/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')

        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.25t/data.txt', dtype=str, delimiter=',')  # DDA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 250/data.txt', dtype=str, delimiter=',')#FBA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 zhengtai/data.txt', dtype=str, delimiter=',')#ADA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')#GWA

        data_pin_normal = np.loadtxt('D:/数据_刘嘉帅/combined_data.txt', dtype=str, delimiter=',')
        data_pin_abnormal_933 = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_caehmm_0.993.txt', dtype=str, delimiter=',')
        data_pin_abnormal_25t = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_caehmm_0.25t.txt', dtype=str, delimiter=',')
        data_pin_abnormal_sin = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_caehmm_sin.txt', dtype=str, delimiter=',')
        data_pin_abnormal_zhengtai = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_caehmm_zhengtai.txt', dtype=str,
                                                delimiter=',')


        # data_r10_normal = np.delete(data_r10_normal, np.array([0, 5, 8]), 1)
        # data_r7dot5_normal = np.delete(data_r7dot5_normal, np.array([0, 5, 8]), 1)
        # data_r5_normal = np.delete(data_r5_normal, np.array([0, 5, 8]), 1)
        # data_r10_abnormal = np.delete(data_r10_abnormal, np.array([0, 5, 8]), 1)
        # data_r7dot5_abnormal = np.delete(data_r7dot5_abnormal, np.array([0, 5, 8]), 1)
        # data_r5_abnormal = np.delete(data_r5_abnormal, np.array([0, 5, 8]), 1)

        data_pin_normal = np.delete(data_pin_normal, np.array([0, 5, 8]), 1)
        data_pin_abnormal_933 = np.delete(data_pin_abnormal_933, np.array([0, 5, 8]), 1)
        data_pin_abnormal_25t = np.delete(data_pin_abnormal_25t, np.array([0, 5, 8]), 1)
        data_pin_abnormal_sin = np.delete(data_pin_abnormal_sin, np.array([0, 5, 8]), 1)
        data_pin_abnormal_zhengtai = np.delete(data_pin_abnormal_zhengtai, np.array([0, 5, 8]), 1)

        _all_length = 160000
        # matrix_nofault = np.concatenate((data_r10_normal[:_all_length, :],
        #                                  data_r7dot5_normal[:_all_length, :],
        #                                  data_r5_normal[:_all_length, :]), axis=0)
        # matrix_addfault = np.concatenate((data_r10_abnormal[:_all_length, :],
        #                                   data_r7dot5_abnormal[:_all_length, :],
        #                                   data_r5_abnormal[:_all_length, :]), axis=0)

        matrix_nofault = data_pin_normal
        matrix_addfault = data_pin_abnormal_933
        # matrix_addfault = data_pin_abnormal_25t
        # matrix_addfault = data_pin_abnormal_sin
        # matrix_addfault = data_pin_abnormal_zhengtai

        # 归一化
        def min_max_normalize_columns(data_column, min_val, max_val):
            data_column = data_column.astype(float)  # Convert data to float type
            return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
                    max_val - min_val) + min_val

        aaaa = _all_length // 2
        slid_window = 400
        matrix_nofault = min_max_normalize_columns(matrix_nofault, 0, 1)
        matrix_addfault = min_max_normalize_columns(matrix_addfault, 0, 1)
        matrix_nofault1 = np.reshape(matrix_nofault, (3, _all_length, -1))
        matrix_nofault2_train = matrix_nofault1[:, :_all_length // 2, :]  # (3,80000,6)
        matrix_nofault2_test = matrix_nofault1[:, _all_length // 2:, :]
        matrix_nofault3_train = np.reshape(matrix_nofault2_train, (240000, -1))
        matrix_nofault3_test = np.reshape(matrix_nofault2_test, (240000, -1))
        matrix_nofault4_train = np.reshape(matrix_nofault3_train, (slid_window, 240000 // slid_window, -1))
        matrix_nofault4_test = np.reshape(matrix_nofault3_test, (slid_window, 240000 // slid_window, -1))

        matrix_nofault5_train = np.transpose(matrix_nofault4_train, (1, 2, 0))  # (600,6,400)
        matrix_nofault5_test = np.transpose(matrix_nofault4_test, (1, 2, 0))  #

        matrix_addfault1 = np.reshape(matrix_addfault, (slid_window, 480000 // slid_window, -1))
        matrix_addfault2 = np.transpose(matrix_addfault1, (1, 2, 0))
        matrix_test = np.vstack((matrix_nofault5_test, matrix_addfault2))
        train_labels = np.vstack((np.zeros((600, 1))))
        test_labels = np.vstack((np.zeros((600, 1)), np.ones((1200, 1))))

        self.train = matrix_nofault5_train
        # self.train = self.train.reshape(-1, 9, 16)
        self.train_labels = train_labels

        self.test = matrix_test
        self.test_labels = test_labels
        # self.test = self.test.reshape(-1, 9, 16)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    dataset = KDD99Loader(data_path, mode)

    shuffle = False #false是正常不打乱，而true是打乱顺序
    #cw
    # if mode == 'train':
    #     shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
