import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from tqdm import tqdm
signal_size = 2048

datasetname = ["1800_0", "1800_0.1", "2400_0", "2400_0.1", "3000_0", "3000_0.1"]  # condition


dataname0 = ["NN_00.mat", "IN_00.mat", "ON_00.mat", "BN_00.mat", "CN_00.mat", "TN_00.mat", "N1_00.mat", "N2_00.mat"]  # 1800rpm_0A 00
# dataname0 = ["NN_00.mat", "IN_00.mat"]  # 1800rpm_0A 00
dataname1 = ["NN_01.mat", "IN_01.mat", "ON_01.mat", "BN_01.mat", "CN_01.mat", "TN_01.mat", "N1_01.mat", "N2_01.mat"]  # 1800rpm_0.1A 00
dataname2 = ["NN_10.mat", "IN_10.mat", "ON_10.mat", "BN_10.mat", "CN_10.mat", "TN_10.mat", "N1_10.mat", "N2_10.mat"]  # 2400rpm_0A 10
dataname3 = ["NN_11.mat", "IN_11.mat", "ON_11.mat", "BN_11.mat", "CN_11.mat", "TN_11.mat", "N1_11.mat", "N2_11.mat"]  # 2400rpm_0.1A 11
dataname4 = ["NN_20.mat", "IN_20.mat", "ON_20.mat", "BN_20.mat", "CN_20.mat", "TN_20.mat", "N1_20.mat", "N2_20.mat"]  # 3000rpm_0A 20
dataname5 = ["NN_21.mat", "IN_21.mat", "ON_21.mat", "BN_21.mat", "CN_21.mat", "TN_21.mat", "N1_21.mat", "N2_21.mat"]  # 3000rpm_0.1A 21
dataname = np.concatenate((dataname0, dataname1, dataname2, dataname3, dataname4, dataname5), axis=0)
# print(dataname[:8])
class AnomalyDetectionDataLoader:
    def __init__(self, data_directory, batch_size=32):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.train_loader = None
        self.test_loader = None

    def load_data(self, file_name):
        data_path = os.path.join(self.data_directory, file_name)
        fl = loadmat(data_path)["data"]
        data = []
        labels = []
        start, end = 0, signal_size
        while end < 2048*101:
            segment = fl[start:end]
            data.append(segment)
            if os.path.basename(data_path).startswith("NN"):
                labels.append(0)
            else:
                labels.append(1)
            start += 1024
            end += 1024
            # start += signal_size
            # end += signal_size
        return data, labels

    def process_data(self):
        all_data = []
        all_labels = []
        for i in range(6):
            root_file = os.path.join(self.data_directory, datasetname[i])
            for filename in tqdm(dataname[8*i: 8*(i+1)]):  # 假设 root 目录下都是 .mat 文件
                # print(filename)
                full_filename = os.path.join(root_file, filename)
                # print(full_filename)
                data_segment, label_segment = self.load_data(full_filename)
                all_data.append(data_segment)
                all_labels.append(label_segment)
        all_data = np.array(all_data).reshape(-1, 1, signal_size)
        all_labels = np.array(all_labels).reshape(-1)

        # Convert to PyTorch tensors
        all_data = torch.tensor(all_data, dtype=torch.float32)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        return all_data, all_labels

    def create_dataloaders(self):
        all_data, all_labels = self.process_data()

        data_0 = all_data[all_labels == 0]
        labels_0 = all_labels[all_labels == 0]
        data_1 = all_data[all_labels == 1]
        labels_1 = all_labels[all_labels == 1]

        # 将标签为0的数据分为两部分，一半用于训练
        train_data_0, test_data_0, train_labels_0, _ = train_test_split(data_0, labels_0, test_size=0.2, random_state=42)
        # 0.2-->240; 0.25-->300; 0.1-->120

        # 将标签为1的数据分为两部分，一半用于测试
        _, test_data_1, _, test_labels_1 = train_test_split(data_1, labels_1, test_size=0.005, random_state=42)
        # 0.1-->840; 0.01-->84; 0.005-->42

        # 组合测试集
        test_data = torch.cat((test_data_0, test_data_1), 0)
        test_labels = torch.cat((torch.zeros(len(test_data_0)), torch.ones(len(test_data_1))), 0)

        # 创建 TensorDataset
        train_dataset = TensorDataset(train_data_0, train_labels_0)
        test_dataset = TensorDataset(test_data, test_labels)

        # # 训练时有异常
        # train_data_0, test_data_0, train_labels_0, _ = train_test_split(data_0, labels_0, test_size=0.2, random_state=42)
        # # 0.2-->960,240; 0.25-->300
        # _, test_data_1, _, test_labels_1 = train_test_split(data_1, labels_1, test_size=0.01, random_state=42)
        # # 0.1-->840; 0.01-->84; 0.05-->420
        # _, train_data_1, _, train_labels_1 = train_test_split(data_1, labels_1, test_size=0.05, random_state=40)
        #
        # # 组合测试集
        # test_data = torch.cat((test_data_0, test_data_1), 0)
        # test_labels = torch.cat((torch.zeros(len(test_data_0)), torch.ones(len(test_data_1))), 0)
        #
        # # 组合训练集
        # train_data = torch.cat((train_data_0, train_data_1), 0)
        # train_labels = torch.cat((torch.zeros(len(train_data_0)), torch.ones(len(train_data_1))), 0)
        #
        # # 创建 TensorDataset
        # train_dataset = TensorDataset(train_data, train_labels)
        # test_dataset = TensorDataset(test_data, test_labels)



        # 创建 DataLoader
        # self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        # 创建 DataLoadercw
        self.train_loader = DataLoader(train_dataset, batch_size=960, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=282, shuffle=True)

    def get_dataloaders(self):
        if not self.train_loader or not self.test_loader:
            self.create_dataloaders()
        return self.train_loader, self.test_loader

# 使用示例:
data_loader = AnomalyDetectionDataLoader(data_directory=r'D:\lzj\48FC')
train_loader, test_loader = data_loader.get_dataloaders()

