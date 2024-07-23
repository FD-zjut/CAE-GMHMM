import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from FC48_Adjust import AnomalyDetectionDataLoader

data = AnomalyDetectionDataLoader(data_directory=r'D:\lzj\48FC')
train_loader, test_loader = data.get_dataloaders()

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size=2048):
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
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型、损失函数和优化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练自编码器
num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    reconstructs = []
    for data in train_loader:
        inputs, labels = data  # 假设每个批次的数据是(inputs, labels)的形式
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        reconstruct = ((outputs - inputs) ** 2).mean(axis=2)
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
    threshold = np.percentile(reconstructs, 95)  # 设置一个合适的阈值
    print("threshold:", threshold)
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            outputs = model(inputs)
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
