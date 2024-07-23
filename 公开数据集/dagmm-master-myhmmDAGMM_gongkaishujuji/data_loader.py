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
import torch.nn.functional as F
import os
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode

        # # CRWU_CW
        # datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data",
        #                "48k Drive End Bearing Fault Data",
        #                "Normal Baseline Data"]
        # normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
        # # For 12k Drive End Bearing Fault Data
        # dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
        #              "234.mat"]  # 1797rpm
        # dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
        #              "235.mat"]  # 1772rpm
        # dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
        #              "236.mat"]  # 1750rpm
        # dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
        #              "237.mat"]  # 1730rpm
        # root = r"D:\cw\CWRU"
        # label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # axis = ["_DE_time", "_FE_time", "_BA_time"]
        # signal_size = 2048
        #
        # def data_load(filename, axisname, label):
        #     '''
        #     This function is mainly used to generate test data and training data.
        #     filename:Data location
        #     axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
        #     '''
        #     datanumber = axisname.split(".")
        #     if eval(datanumber[0]) < 100:
        #         realaxis = "X0" + datanumber[0] + axis[0]
        #     else:
        #         realaxis = "X" + datanumber[0] + axis[0]
        #     fl = loadmat(filename)[realaxis]
        #     data = []
        #     lab = []
        #     start, end = 0, signal_size
        #     while end <= fl.shape[0]:
        #         data.append(fl[start:end])
        #         lab.append(label)
        #         start += signal_size
        #         end += signal_size
        #
        #     return data, lab
        #
        # data_root1 = os.path.join('/tmp', root, datasetname[3])
        # data_root2 = os.path.join('/tmp', root, datasetname[0])
        #
        # # 正常数据
        # path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
        # data_normal, lab_normal = data_load(path1, axisname=normalname[0], label=0)  # nThe label for normal data is 0
        # for m in tqdm(range(1, 4)):
        #     path1 = os.path.join('/tmp', data_root1, normalname[m])
        #     # data1, lab1 = data_load(path1, normalname[m], label=label[m - 1])
        #     data1, lab1 = data_load(path1, normalname[m], label=0)
        #     data_normal += data1
        #     lab_normal += lab1
        #
        # # 故障数据
        # data_abnormal1 = []
        # lab_abnormal1 = []
        # path2 = os.path.join('/tmp', data_root2, dataname1[0])
        # data_abnormal1_cond1, lab_abnormal1_cond1 = data_load(path2, axisname=dataname1[0], label=1)
        # path2 = os.path.join('/tmp', data_root2, dataname2[0])
        # data_abnormal1_cond2, lab_abnormal1_cond2 = data_load(path2, axisname=dataname2[0], label=1)
        # path2 = os.path.join('/tmp', data_root2, dataname3[0])
        # data_abnormal1_cond3, lab_abnormal1_cond3 = data_load(path2, axisname=dataname3[0], label=1)
        # path2 = os.path.join('/tmp', data_root2, dataname4[0])
        # data_abnormal1_cond4, lab_abnormal1_cond4 = data_load(path2, axisname=dataname4[0], label=1)
        # data_abnormal1 += data_abnormal1_cond1
        # data_abnormal1 += data_abnormal1_cond2
        # data_abnormal1 += data_abnormal1_cond3
        # data_abnormal1 += data_abnormal1_cond4
        # lab_abnormal1 += lab_abnormal1_cond1
        # lab_abnormal1 += lab_abnormal1_cond2
        # lab_abnormal1 += lab_abnormal1_cond3
        # lab_abnormal1 += lab_abnormal1_cond4
        # lab_abnormal1 = np.array(lab_abnormal1)
        # lab_normal = np.array(lab_normal)
        # train_pd, val_pd = train_test_split(data_normal, test_size=0.202, random_state=4)
        # X_test = val_pd + data_abnormal1
        # X_test = np.array(X_test)
        # data_abnormal1 = np.array(data_abnormal1)
        # selftrain = np.array(train_pd)
        # selftrain_labels = lab_normal[:len(selftrain)]
        # selftest_labels = lab_normal[len(selftrain):]
        # selftest_labels = np.concatenate((selftest_labels, lab_abnormal1), axis=0)
        # selftrain = np.reshape(selftrain, (-1, 1, 2048))
        # X_test = np.reshape(X_test, (-1, 1, 2048))
        #
        # self.train = selftrain
        # self.train_labels = selftrain_labels
        #
        # self.test_labels = selftest_labels
        #
        # self.test = X_test
        # print("1")

        #lzj###########################
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



        # cw 由matlab代码转换
        # 收集采集pos、速度和力矩数据
        data_normal = np.loadtxt('D:/变工况数据集/实验用/正常样本/normal_speed.txt', dtype=str, delimiter=',')
        # data_abnormal = np.loadtxt(data_path, dtype=str,  delimiter=',')  # data: (200, 1601)

        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.02sin(2pi-200)\\data.txt', dtype=str, delimiter=',')#均为1
        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')#auc: 0.9925, ap: 0.9950, F1 Score: 0.9924
        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')#均为1

        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 1000sin(2pi-100)\\data.txt', dtype=str, delimiter=',')#均为1
        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 1000sin(2pi-200)\\data.txt', dtype=str, delimiter=',')#均为1
        # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 1000sin(2pi-400)\\data.txt', dtype=str, delimiter=',')#均为1
        # data_abnormal = np.loadtxt('D:/变工况数据集/实验用/异常样本/abnormal_pitting.txt', dtype=str,
        #                                           delimiter=',')#auc: 0.9287, ap: 0.9525, F1 Score: 0.9233
        # data_abnormal = np.loadtxt('D:/变工况数据集/实验用/异常样本/abnormal_break.txt', dtype=str,
        #                                         delimiter=',')
        # data_abnormal = np.loadtxt('D:/变工况数据集/实验用/异常样本/abnormal_crack.txt', dtype=str,
        #                                         delimiter=',')
        data_abnormal = np.loadtxt('D:/变工况数据集/实验用/异常样本/abnormal_wear.txt', dtype=str,
                                               delimiter=',')

        silde_window = 1600  # 400（1） #800（0）#1600（1）
        all_length = 160000
        data_length = np.int(all_length / silde_window)
        data_nofault_pos = data_normal[:all_length, 2]
        spe_x1 = data_normal[:all_length, 1]
        liju_x1 = data_normal[:all_length, 6]

        data_GWF_pos = data_abnormal[:all_length, 2]
        spe_x2 = data_abnormal[:all_length, 1]
        liju_x2 = data_abnormal[:all_length, 6]
        # 归一化*100*
        def min_max_normalize_columns(data_column, min_val, max_val):
            data_column = data_column.astype(float)  # Convert data to float type
            return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
                        max_val - min_val) + min_val

        data_nofault_pos = min_max_normalize_columns(data_nofault_pos, 0, 1)
        spe_x1 = min_max_normalize_columns(spe_x1, 0, 1)
        liju_x1 = min_max_normalize_columns(liju_x1, 0, 1)
        data_GWF_pos = min_max_normalize_columns(data_GWF_pos, 0, 1)
        spe_x2 = min_max_normalize_columns(spe_x2, 0, 1)
        liju_x2 = min_max_normalize_columns(liju_x2, 0, 1)
        # 再合并速度、位置、力矩
        # 将数据重新形状为100行、1600列（100，1600）

        reshaped_data_nofault_pos = np.reshape(data_nofault_pos, (data_length, silde_window))
        reshaped_data_nofault_spe = np.reshape(spe_x1, (data_length, silde_window))
        reshaped_data_nofault_liju = np.reshape(liju_x1, (data_length, silde_window))
        reshaped_data_GWF_pos = np.reshape(data_GWF_pos, (data_length, silde_window))
        reshaped_data_GWF_spe = np.reshape(spe_x2, (data_length, silde_window))
        reshaped_data_GWF_liju = np.reshape(liju_x2, (data_length, silde_window))

        # 将三个矩阵连接起来
        matrix_nofault = np.concatenate(
            (reshaped_data_nofault_pos, reshaped_data_nofault_spe, reshaped_data_nofault_liju), axis=1)
        matrix_data_GWF = np.concatenate((reshaped_data_GWF_pos, reshaped_data_GWF_spe, reshaped_data_GWF_liju), axis=1)

        matrix_nofault= np.hstack((matrix_nofault, np.zeros((data_length,1))))
        matrix_data_GWF = np.hstack((matrix_data_GWF, np.ones((data_length,1))))
        # 使用 vstack 将两个矩阵垂直拼接在一起
        merged_data = np.vstack((matrix_nofault, matrix_data_GWF))
        data = merged_data

        # #cw
        # data = np.loadtxt(data_path, delimiter=',')#data: (200, 1601)
        labels = data[:, -1]  # 取最后一列数据
        print("labels:", labels.shape)#labels: (200,)
        features = data[:,:-1]#取除最后一列外的所有列 #(200, 1600)

        N, D = features.shape

        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        print("randIdx", randIdx,randIdx.shape)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        self.train = attack_data[randIdx[:N_train]]
        self.train = self.train.reshape(-1,3,silde_window)
        self.train_labels = attack_labels[randIdx[:N_train]]

        self.test = attack_data[randIdx[N_train:]]
        self.test_labels = attack_labels[randIdx[N_train:]]

        self.test = np.concatenate((self.test, normal_data),axis=0)
        self.test = self.test.reshape(-1, 3, silde_window)

        # original_array = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
        # # 使用reshape函数将其转换为（50，3，1600）的形状
        # new_array = original_array.reshape((3, 2, 3))
        # print("new_array",new_array)

        self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)
        # self.test = np.concatenate((self.test, normal_data[0:50]), axis=0)
        # self.test_labels = np.concatenate((self.test_labels, normal_labels[0:50]), axis=0)
        # print("self.test_labels",self.test_labels)
        # print(self.train.shape)


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
