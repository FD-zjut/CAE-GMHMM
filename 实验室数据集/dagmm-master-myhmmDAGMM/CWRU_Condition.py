import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from SequenceDatasets import dataset
from sequence_aug import *
from tqdm import tqdm
import torch

signal_size = 2048


datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
               "Normal Baseline Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
dataname1 = ["105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
             "234.mat"]  # 1797rpm
dataname2 = ["106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
             "235.mat"]  # 1772rpm
dataname3 = ["107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
             "236.mat"]  # 1750rpm
dataname4 = ["108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
             "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
dataname5 = ["278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
             "315.mat"]  # 1797rpm
dataname6 = ["279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
             "316.mat"]  # 1772rpm
dataname7 = ["280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
             "317.mat"]  # 1750rpm
dataname8 = ["281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
             "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
dataname9 = ["109.mat", "122.mat", "135.mat", "174.mat", "189.mat", "201.mat", "213.mat", "250.mat",
             "262.mat"]  # 1797rpm
dataname10 = ["110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "251.mat",
              "263.mat"]  # 1772rpm
dataname11 = ["111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "252.mat",
              "264.mat"]  # 1750rpm
dataname12 = ["112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "253.mat",
              "265.mat"]  # 1730rpm
# label
# The failure data is labeled 1-9
# label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,\
#          25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
# label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
axis = ["_DE_time", "_FE_time", "_BA_time"]


# generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    data_root1 = os.path.join('/tmp', root, datasetname[3])
    data_root2 = os.path.join('/tmp', root, datasetname[0])


    path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
    data, lab = data_load(path1, axisname=normalname[0], label=0)  # nThe label for normal data is 0
    for m in tqdm(range(1, 4)):
        path1 = os.path.join('/tmp', data_root1, normalname[m])
        data1, lab1 = data_load(path1, normalname[m], label=label[m-1])
        data += data1
        lab += lab1

    for i in tqdm(range(len(dataname1))):
        path2 = os.path.join('/tmp', data_root2, dataname1[i])

        data1, lab1 = data_load(path2, dataname1[i], label=0)
        data += data1
        lab += lab1
    for j in tqdm(range(len(dataname2))):
        path2 = os.path.join('/tmp', data_root2, dataname2[j])

        data1, lab1 = data_load(path2, dataname2[j], label=1)
        data += data1
        lab += lab1
    for k in tqdm(range(len(dataname3))):
        path2 = os.path.join('/tmp', data_root2, dataname3[k])

        data1, lab1 = data_load(path2, dataname3[k], label=2)
        data += data1
        lab += lab1
    for l in tqdm(range(len(dataname4))):
        path2 = os.path.join('/tmp', data_root2, dataname4[l])

        data1, lab1 = data_load(path2, dataname4[l], label=3)
        data += data1
        lab += lab1

    return [data, lab]


def data_load(filename, axisname, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        realaxis = "X0" + datanumber[0] + axis[0]
    else:
        realaxis = "X" + datanumber[0] + axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


# def data_transforms(dataset_type="train", normlize_type="-1-1"):
#     transforms = {
#         'train': Compose([
#             Reshape(),
#             Normalize(normlize_type),
#             RandomAddGaussian(),
#             RandomScale(),
#             RandomStretch(),
#             RandomCrop(),
#             Retype()
#
#         ]),
#         'val': Compose([
#             Reshape(),
#             Normalize(normlize_type),
#             Retype()
#         ])
#     }
#     return transforms[dataset_type]

class CWRU_Condition(object):
    # num_classes = 10
    inputchannel = 1

    def __init__(self, data_dir, normlizetype):
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):

        list_data = get_files(self.data_dir, test)
        if test:
            test_dataset = dataset(list_data=list_data, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.20, random_state=4, stratify=data_pd["label"])
            # train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            # val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            train_dataset = dataset(list_data=train_pd, transform=None)
            data_train_0_all = []
            # # cw
            # for data in train_dataset:
            #     train_inputs, train_labels = data
            #     data_train_0 = train_inputs[train_labels == 0]
            #     data_train_0_all.append(data_train_0)
            val_dataset = dataset(list_data=val_pd, transform=None)
            return train_dataset, val_dataset

class CWRU_Condition_Dataloaders():
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
    def get_Dataloaders(self):
        datasets = {}

        datasets['train'], datasets['val'] = CWRU_Condition(r"D:\cw\CWRU",
                                                           normlizetype="1-1").data_preprare()
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=self.batch_size,
                                                      shuffle=False,
                                                      pin_memory=(True if self.device == 'cuda' else False), drop_last=True)
                       for x in ['train', 'val']}
        # return dataloaders
        return dataloaders['train'], dataloaders['val']