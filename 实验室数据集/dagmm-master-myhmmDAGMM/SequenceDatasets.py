import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from sequence_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        # if self.test:  # 如果是测试集，则只包含数据，没用标签
        #     self.seq_data = list_data['data'].tolist()
        # else:  # 如果是训练集，则同时包括数据和标签
        #     self.seq_data = list_data['data'].tolist()
        #     self.labels = list_data['label'].tolist()
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        if transform is None:  # 如果未指定数据增强函数，则使用 Reshape 转换
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform

    # 获取数据集中样本的数量
    def __len__(self):
        return len(self.seq_data)

    # 获取指定索引处的数据和标签
    def __getitem__(self, item):
        if self.test:  # 如果是测试集，只返回数据和索引
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:  # 如果是训练集，则同时返回数据和标签
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label

