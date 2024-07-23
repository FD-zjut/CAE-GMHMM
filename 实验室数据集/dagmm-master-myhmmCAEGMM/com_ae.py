import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
data_abnormal = np.loadtxt('./data.txt', dtype=str, delimiter=',')  # data: (200, 1601)
# data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')
data_nofault_pos = data_normal[:160000, 2]
spe_x1 = data_normal[:160000, 1]
liju_x1 = data_normal[:160000, 6]

data_GWF_pos = data_abnormal[:160000, 2]
spe_x2 = data_abnormal[:160000, 1]
liju_x2 = data_abnormal[:160000, 6]

# 归一化
def min_max_normalize_columns(data_column, min_val, max_val):
    data_column = data_column.astype(float)  # Convert data to float type
    return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
            max_val - min_val) + min_val

# data_nofault_pos = min_max_normalize_columns(data_nofault_pos, 0, 1)
# spe_x1 = min_max_normalize_columns(spe_x1, 0, 1)
# liju_x1 = min_max_normalize_columns(liju_x1, 0, 1)
# data_GWF_pos = min_max_normalize_columns(data_GWF_pos, 0, 1)
# spe_x2 = min_max_normalize_columns(spe_x2, 0, 1)
# liju_x2 = min_max_normalize_columns(liju_x2, 0, 1)

data_nofault_pos = data_nofault_pos.astype(float)
spe_x1 = spe_x1.astype(float)
liju_x1 = liju_x1.astype(float)
data_GWF_pos = data_GWF_pos.astype(float)
spe_x2 = spe_x2.astype(float)
liju_x2 = liju_x2.astype(float)

# 再合并速度、位置、力矩
# 将数据重新形状为100行、1600列
all_length = 160000
silde_window = 200
data_length = np.int(all_length/silde_window)
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

matrix_nofault = np.hstack((matrix_nofault, np.zeros((data_length, 1))))
matrix_data_GWF = np.hstack((matrix_data_GWF, np.ones((data_length, 1))))
# # 使用 vstack 将两个矩阵垂直拼接在一起
merged_data = np.vstack((matrix_nofault, matrix_data_GWF))
data = merged_data

# #cw
# data = np.loadtxt(data_path, delimiter=',')#data: (200, 1601)
labels = data[:, -1]  # 取最后一列数据
# print("labels:", labels.shape)  # labels: (200,)
features = data[:, :-1]  # 取除最后一列外的所有列 #(200, 1600)

N, D = features.shape
normal_data = features[labels == 1]
normal_labels = labels[labels == 1]

N_normal = normal_data.shape[0]

attack_data = features[labels == 0]
attack_labels = labels[labels == 0]

N_attack = attack_data.shape[0]

randIdx = np.arange(N_attack)
# print("randIdx", randIdx, randIdx.shape)
np.random.shuffle(randIdx)
N_train = N_attack // 2

selftrain = attack_data[randIdx[:N_train]]
# selftrain = selftrain.reshape(-1, 3, 400)
selftrain = selftrain.reshape(-1, 3*silde_window)
selftrain_labels = attack_labels[randIdx[:N_train]]

selftest = attack_data[randIdx[N_train:]]
selftest_labels = attack_labels[randIdx[N_train:]]

selftest = np.concatenate((selftest, normal_data), axis=0)
# selftest = selftest.reshape(-1, 3, 400)
selftest = selftest.reshape(-1, 3*silde_window)
selftest_labels = np.concatenate((selftest_labels, normal_labels), axis=0)

matrix_nofault5_train = torch.tensor(selftrain, dtype=torch.float32)
matrix_test = torch.tensor(selftest, dtype=torch.float32)
train_labels = torch.tensor(selftrain_labels, dtype=torch.long)
test_labels = torch.tensor(selftest_labels, dtype=torch.long)

# 创建 TensorDataset
train_dataset = TensorDataset(matrix_nofault5_train, train_labels)
test_dataset = TensorDataset(matrix_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size=3*silde_window):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型、损失函数和优化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) #-4

# 训练自编码器
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    reconstructs = []
    for data in train_loader:
        inputs, labels = data  # 假设每个批次的数据是(inputs, labels)的形式
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
            outputs = model(inputs)
            # loss = ((outputs - inputs) ** 2).mean(axis=2)#CW?
            loss = ((outputs - inputs) ** 2).mean(axis=1)
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
