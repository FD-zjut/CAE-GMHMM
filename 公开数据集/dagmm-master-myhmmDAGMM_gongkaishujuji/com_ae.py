import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from CWRU_Condition import CWRU_Condition_Dataloaders
import torch.nn.functional as F
import os
from scipy.io import loadmat
from tqdm import tqdm
from SequenceDatasets import dataset
#CRWU_CW
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
root = r"D:\cw\CWRU"
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
axis = ["_DE_time", "_FE_time", "_BA_time"]
signal_size = 2048
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
data_root1 = os.path.join('/tmp', root, datasetname[3])
data_root2 = os.path.join('/tmp', root, datasetname[0])

#正常数据
path1 = os.path.join('/tmp', data_root1, normalname[0])  # 0->1797rpm ;1->1772rpm;2->1750rpm;3->1730rpm
data_normal, lab_normal = data_load(path1, axisname=normalname[0], label=0)  # nThe label for normal data is 0
for m in tqdm(range(1, 4)):
    path1 = os.path.join('/tmp', data_root1, normalname[m])
    # data1, lab1 = data_load(path1, normalname[m], label=label[m - 1])
    data1, lab1 = data_load(path1, normalname[m], label=0)
    data_normal += data1
    lab_normal += lab1

#故障数据
data_abnormal1 = []
lab_abnormal1 = []
path2 = os.path.join('/tmp', data_root2, dataname1[0])
data_abnormal1_cond1, lab_abnormal1_cond1 = data_load(path2, axisname=dataname1[0], label=1)
path2 = os.path.join('/tmp', data_root2, dataname2[0])
data_abnormal1_cond2, lab_abnormal1_cond2 = data_load(path2, axisname=dataname2[0], label=1)
path2 = os.path.join('/tmp', data_root2, dataname3[0])
data_abnormal1_cond3, lab_abnormal1_cond3 = data_load(path2, axisname=dataname3[0], label=1)
path2 = os.path.join('/tmp', data_root2, dataname4[0])
data_abnormal1_cond4, lab_abnormal1_cond4 = data_load(path2, axisname=dataname4[0], label=1)
data_abnormal1 += data_abnormal1_cond1
data_abnormal1 += data_abnormal1_cond2
data_abnormal1 += data_abnormal1_cond3
data_abnormal1 += data_abnormal1_cond4
lab_abnormal1 += lab_abnormal1_cond1
lab_abnormal1 += lab_abnormal1_cond2
lab_abnormal1 += lab_abnormal1_cond3
lab_abnormal1 += lab_abnormal1_cond4
lab_abnormal1 = np.array(lab_abnormal1)
lab_normal = np.array(lab_normal)
train_pd, val_pd = train_test_split(data_normal, test_size=0.20, random_state=4)
X_test = val_pd + data_abnormal1
X_test = np.array(X_test)
data_abnormal1 = np.array(data_abnormal1)
selftrain = np.array(train_pd)
selftrain_labels = lab_normal[:len(selftrain)]
selftest_labels = lab_normal[len(selftrain):]
selftest_labels = np.concatenate((selftest_labels, lab_abnormal1),axis=0)
selftrain = np.reshape(selftrain, (-1, 1, 2048))
X_test = np.reshape(X_test, (-1, 1, 2048))
# 创建 TensorDataset
train_dataset = TensorDataset(torch.tensor(selftrain), torch.tensor(selftrain_labels))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(selftest_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# #CRWU
# datasets = {}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# datasets['train'], datasets['val'] = CWRU_Condition_Dataloaders(batch_size=20, device=device).get_Dataloaders()
# # data_train_0_all = []
# current_row = 0
# total_x = 0
# count = 0
# data_test_all = []
# label_test_all = []
# for data in datasets['train']:
#     train_inputs, train_labels = data
#     # 使用 torch.eq 将张量与零进行逐元素比较
#     zero_mask = torch.eq(train_labels, 0)
#     # 使用 torch.sum 对比较结果进行求和，得到零的数量
#     num_zeros = torch.sum(zero_mask).item()
#     total_x = total_x + num_zeros
# result_tensor = torch.zeros(total_x, 1, 2048)
# # selftrain_labels = np.zeros((total_x, 1))
# selftrain_labels = torch.zeros((total_x))
#
# for data in datasets['train']:
#     train_inputs, train_labels = data
#     data_train_0 = train_inputs[train_labels == 0]
#     rows = data_train_0.shape[0]
#     result_tensor[current_row:current_row + rows] = data_train_0
#     current_row += rows
#
# for data in datasets['val']:
#     test_inputs, test_labels = data
#     # data_test_not0 = test_inputs[test_labels != 0]
#     # data_not0_all.append(data_test_not0)
#     data_test_all.append(test_inputs)
#     label_test_all.append(test_labels)
#     count += 1
# # print(count)  # 29
# result_data_test_all = torch.cat(data_test_all, dim=0)
# result_label_test_all = torch.cat(label_test_all, dim=0)
#
# # D1wei, _, D2wei = result_tensor.size()
# # X_train = (result_tensor.view(D1wei, D2wei)).numpy()
# # X_train = result_tensor.view(D1wei, D2wei)
# X_train = result_tensor
#
# selftrain = X_train
# # selftrain_labels = selftrain_labels.reshape(-1)
#
# result_label_test_all[result_label_test_all != 0] = 1
# # selftest_labels = result_label_test_all.numpy()
# selftest_labels = result_label_test_all
#
# # D1wei, _, D2wei = result_data_test_all.size()
# # X_test = (result_data_test_all.view(D1wei, D2wei)).numpy()
# # X_test = result_data_test_all.view(D1wei, D2wei)
# X_test = result_data_test_all
# selftest = X_test
#
# # 创建 TensorDataset
# train_dataset = TensorDataset(selftrain, selftrain_labels)
# test_dataset = TensorDataset(X_test, selftest_labels)
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# print("1")




# data_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
# data_abnormal = np.loadtxt('./data.txt', dtype=str, delimiter=',')  # data: (200, 1601)
# # data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')
# data_nofault_pos = data_normal[:160000, 2]
# spe_x1 = data_normal[:160000, 1]
# liju_x1 = data_normal[:160000, 6]
#
# data_GWF_pos = data_abnormal[:160000, 2]
# spe_x2 = data_abnormal[:160000, 1]
# liju_x2 = data_abnormal[:160000, 6]
#
# # 归一化
# def min_max_normalize_columns(data_column, min_val, max_val):
#     data_column = data_column.astype(float)  # Convert data to float type
#     return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
#             max_val - min_val) + min_val
#
# # data_nofault_pos = min_max_normalize_columns(data_nofault_pos, 0, 1)
# # spe_x1 = min_max_normalize_columns(spe_x1, 0, 1)
# # liju_x1 = min_max_normalize_columns(liju_x1, 0, 1)
# # data_GWF_pos = min_max_normalize_columns(data_GWF_pos, 0, 1)
# # spe_x2 = min_max_normalize_columns(spe_x2, 0, 1)
# # liju_x2 = min_max_normalize_columns(liju_x2, 0, 1)
#
# data_nofault_pos = data_nofault_pos.astype(float)
# spe_x1 = spe_x1.astype(float)
# liju_x1 = liju_x1.astype(float)
# data_GWF_pos = data_GWF_pos.astype(float)
# spe_x2 = spe_x2.astype(float)
# liju_x2 = liju_x2.astype(float)
#
# # 再合并速度、位置、力矩
# # 将数据重新形状为100行、1600列
# all_length = 160000
# silde_window = 200
# data_length = np.int(all_length/silde_window)
# reshaped_data_nofault_pos = np.reshape(data_nofault_pos, (data_length, silde_window))
# reshaped_data_nofault_spe = np.reshape(spe_x1, (data_length, silde_window))
# reshaped_data_nofault_liju = np.reshape(liju_x1, (data_length, silde_window))
# reshaped_data_GWF_pos = np.reshape(data_GWF_pos, (data_length, silde_window))
# reshaped_data_GWF_spe = np.reshape(spe_x2, (data_length, silde_window))
# reshaped_data_GWF_liju = np.reshape(liju_x2, (data_length, silde_window))
# # 将三个矩阵连接起来
# matrix_nofault = np.concatenate(
#     (reshaped_data_nofault_pos, reshaped_data_nofault_spe, reshaped_data_nofault_liju), axis=1)
# matrix_data_GWF = np.concatenate((reshaped_data_GWF_pos, reshaped_data_GWF_spe, reshaped_data_GWF_liju), axis=1)
#
# matrix_nofault = np.hstack((matrix_nofault, np.zeros((data_length, 1))))
# matrix_data_GWF = np.hstack((matrix_data_GWF, np.ones((data_length, 1))))
# # # 使用 vstack 将两个矩阵垂直拼接在一起
# merged_data = np.vstack((matrix_nofault, matrix_data_GWF))
# data = merged_data
#
# # #cw
# # data = np.loadtxt(data_path, delimiter=',')#data: (200, 1601)
# labels = data[:, -1]  # 取最后一列数据
# # print("labels:", labels.shape)  # labels: (200,)
# features = data[:, :-1]  # 取除最后一列外的所有列 #(200, 1600)
#
# N, D = features.shape
# normal_data = features[labels == 1]
# normal_labels = labels[labels == 1]
#
# N_normal = normal_data.shape[0]
#
# attack_data = features[labels == 0]
# attack_labels = labels[labels == 0]
#
# N_attack = attack_data.shape[0]
#
# randIdx = np.arange(N_attack)
# # print("randIdx", randIdx, randIdx.shape)
# np.random.shuffle(randIdx)
# N_train = N_attack // 2
#
# selftrain = attack_data[randIdx[:N_train]]
# # selftrain = selftrain.reshape(-1, 3, 400)
# selftrain = selftrain.reshape(-1, 3*silde_window)
# selftrain_labels = attack_labels[randIdx[:N_train]]
#
# selftest = attack_data[randIdx[N_train:]]
# selftest_labels = attack_labels[randIdx[N_train:]]
#
# selftest = np.concatenate((selftest, normal_data), axis=0)
# # selftest = selftest.reshape(-1, 3, 400)
# selftest = selftest.reshape(-1, 3*silde_window)
# selftest_labels = np.concatenate((selftest_labels, normal_labels), axis=0)
#
# matrix_nofault5_train = torch.tensor(selftrain, dtype=torch.float32)
# matrix_test = torch.tensor(selftest, dtype=torch.float32)
# train_labels = torch.tensor(selftrain_labels, dtype=torch.long)
# test_labels = torch.tensor(selftest_labels, dtype=torch.long)
#
# # 创建 TensorDataset
# train_dataset = TensorDataset(matrix_nofault5_train, train_labels)
# test_dataset = TensorDataset(matrix_test, test_labels)
#
# train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * np.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))

    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Laplace_fast, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=2, dilation=1, bias=None, groups=1)

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size=2048):
        super(Autoencoder, self).__init__()

        # cw
        # Convolutional layers
        self.layer1 = nn.Sequential(
            Laplace_fast(4, 5),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2))
        # self.Linear1 = nn.Linear(2048, 256)
        # self.Linear2 = nn.Linear(256, 10 * 6)
        # self.Linear3 = nn.Linear(10 * 6, num_states)
        self.Linear1 = nn.Linear(input_size, 150)  # 3*window/2;  window = window/2/2/2 (192=3*128/2) #600
        self.Linear2 = nn.Linear(150, 100)
        self.Linear3 = nn.Linear(100, 50)
        self.Linear4 = nn.Linear(50, 30)
        self.Linear5 = nn.Linear(30, 10)
        # self.Linear6 = nn.Linear(10, latent_dim - 3 * 2)

        # self.fc1 = nn.Linear(latent_dim - 3 * 2, 10)  # map z_c to a higher dimensional space
        self.fc2 = nn.Linear(10, 30)
        self.fc3 = nn.Linear(30, 50)
        self.fc4 = nn.Linear(50, 100)
        self.fc5 = nn.Linear(100, 150)
        self.fc6 = nn.Linear(150, input_size)
        self.layer1_d = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU())
        self.layer2_d = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
        self.layer3_d = nn.Sequential(
            nn.ConvTranspose1d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU())
        self.layer4_d = nn.Sequential(
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)

    def encoder(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = h.view(h.size(0), -1)
        h = self.Linear1(h)
        h = nn.ReLU()(h)
        h = self.Linear2(h)
        h = nn.ReLU()(h)
        h = self.Linear3(h)
        h = nn.ReLU()(h)
        h = self.Linear4(h)
        h = nn.ReLU()(h)
        h = self.Linear5(h)
        h = nn.ReLU()(h)
        # h = self.Linear6(h)
        # h = nn.ReLU()(h)
        # h = self.dropout(h)
        return h

    def decoder(self, x):
        # x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        # print(x.shape)
        x = x.reshape(-1, 16, 128)  # 12通道数
        h = self.layer1_d(x)
        h = self.layer2_d(h)
        h = self.layer3_d(h)
        h = self.layer4_d(h)
        return h

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 32)
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, input_size),
        #     # nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) #-4
model = model.to(device)
# 训练自编码器
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    reconstructs = []
    for data in train_loader:
        inputs, labels = data  # 假设每个批次的数据是(inputs, labels)的形式
        inputs = inputs.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        # a= (outputs - inputs) ** 2
        reconstruct = ((outputs - inputs) ** 2).mean(axis=1)
        # reconstruct = ((outputs - inputs) ** 2).mean(axis=2)#CW?
        reconstructs.append(reconstruct.detach().cpu().numpy())
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    reconstructs = np.array([tensor for tensor in reconstructs], dtype=object)
    reconstructs = np.concatenate(reconstructs, axis=0)


    # 预测和评估
    model.eval()
    all_loss = []
    all_label = []
    threshold = np.percentile(reconstructs, 90)  # 设置一个合适的阈值
    print("threshold:", threshold)
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            inputs = inputs.float().to(device)
            label = label.to(device)
            outputs = model(inputs)
            # loss = ((outputs - inputs) ** 2).mean(axis=2)#CW?
            loss = ((outputs - inputs) ** 2).mean(axis=2)
            # loss = criterion(outputs, inputs)
            # pred = (loss > threshold).type(torch.int).numpy()
            all_loss.append(loss)
            all_label.append(label)
    all_loss = np.array([tensor.cpu().numpy() for tensor in all_loss], dtype=object)
    all_label = np.array([tensor.cpu().numpy() for tensor in all_label], dtype=object)
    all_loss = np.concatenate(all_loss, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    pred = (all_loss > threshold).astype(int)
    gt = all_label.astype(int)
    # 计算指标
    precision_ae, recall_ae, f_score_ae, _ = prf(gt, pred, average='binary', zero_division=1)
    accuracy_ae = accuracy_score(gt, pred)

    print(f"AE - Precision: {precision_ae:.4f}, Recall: {recall_ae:.4f}, F1 Score: {f_score_ae:.4f}, Accuracy: {accuracy_ae:.4f}")
