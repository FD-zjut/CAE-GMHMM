# unsupervised methods
from deepod.models.tabular import RCA, DeepSVDD, REPEN, GOAD, NeuTraL, RDP, ICL, SLAD, DeepIsolationForest
from deepod.models.time_series import DeepIsolationForestTS, DeepSVDDTS, TranAD, USAD, COUTA, AnomalyTransformer, TimesNet
import numpy as np
import torch
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
# from FC48_Adjust import AnomalyDetectionDataLoader
from CWRU_Condition import CWRU_Condition_Dataloaders



# ######################CRWU
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
# selftrain_labels = np.zeros((total_x, 1))
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
# D1wei, _, D2wei = result_tensor.size()
# X_train = (result_tensor.view(D1wei, D2wei)).numpy()
#
# selftrain = X_train
# selftrain_labels = selftrain_labels
#
# result_label_test_all[result_label_test_all != 0] = 1
# selftest_labels = result_label_test_all.numpy()
#
# D1wei, _, D2wei = result_data_test_all.size()
# X_test = (result_data_test_all.view(D1wei, D2wei)).numpy()
#
# selftest = X_test
# print("1")


##########lzj_data
# data = AnomalyDetectionDataLoader(data_directory=r'D:\lzj\48FC')
# train_loader, test_loader = data.get_dataloaders()
# for data in train_loader:
#         inputs, labels = data  # 假设每个批次的数据是(inputs, labels)的形式
# for data1 in test_loader:
#         inputs1, label1 = data1
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
# print("1")



# data_normal800 = pd.read_csv('./电信号故障数据裁剪/NN800.csv', dtype=str, delimiter=',')
# data_normal1200 = pd.read_csv('./电信号故障数据裁剪/NN1200.csv', dtype=str, delimiter=',')
# data_normal1600 = pd.read_csv('./电信号故障数据裁剪/NN1600.csv', dtype=str, delimiter=',')
# data_abnormal800 = pd.read_csv('./电信号故障数据裁剪/BN800.csv', dtype=str, delimiter=',')
# data_abnormal1200 = pd.read_csv('./电信号故障数据裁剪/BN1200.csv', dtype=str, delimiter=',')
# data_abnormal1600 = pd.read_csv('./电信号故障数据裁剪/BN1600.csv', dtype=str, delimiter=',')
# _all_length = 1440
# matrix_nofault = np.concatenate((data_normal800.head(_all_length).values, data_normal1200.head(_all_length).values, data_normal1600.head(_all_length).values), axis=0)
# matrix_addfault = np.concatenate((data_abnormal800.head(_all_length).values, data_abnormal1200.head(_all_length).values, data_abnormal1600.head(_all_length).values), axis=0)
# # 归一化
# def min_max_normalize_columns(data_column, min_val, max_val):
#     data_column = data_column.astype(float)  # Convert data to float type
#     return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
#             max_val - min_val) + min_val
# matrix_nofault = min_max_normalize_columns(matrix_nofault, 0, 1)
# matrix_addfault = min_max_normalize_columns(matrix_addfault, 0, 1)
# matrix_nofault1 = np.reshape(matrix_nofault, (3, _all_length, -1))
# matrix_nofault2 = np.reshape(matrix_nofault1, (3, 16, 90, -1))
# matrix_nofault2_train = matrix_nofault2[:, :, :45, :]
# matrix_nofault2_test = matrix_nofault2[:, :, 45:, :]
# matrix_nofault3_train = np.reshape(matrix_nofault2_train, (16, 135, 9))
# matrix_nofault3_test = np.reshape(matrix_nofault2_test, (16, 135, 9))
# matrix_nofault4_train = np.transpose(matrix_nofault3_train, (1, 0, 2)) #(135,16,9)
# matrix_nofault4_test = np.transpose(matrix_nofault3_test, (1, 0, 2))  # (135,16,9)
# matrix_nofault5_train = np.reshape(matrix_nofault4_train, (-1, 144)) #(135,144)
# matrix_nofault5_test = np.reshape(matrix_nofault4_test, (-1, 144))  # (135,144)
# matrix_addfault1 = np.reshape(matrix_addfault, (3, _all_length, -1))
# matrix_addfault2 = np.reshape(matrix_addfault1, (3, 16, 90, -1))
# matrix_addfault3 = np.reshape(matrix_addfault2, (16, 270, 9))
# matrix_addfault4 = np.transpose(matrix_addfault3, (1, 0, 2))#(900,6,10)
# matrix_addfault5 = np.reshape(matrix_addfault4, (-1, 144)) #(900,60)
# matrix_test =  np.vstack((matrix_nofault5_test, matrix_addfault5))
# train_labels = np.vstack((np.zeros((135, 1))))
# test_labels =  np.vstack((np.zeros((135, 1)),np.ones((270, 1))))

# selftrain = matrix_nofault5_train
# selftrain_labels = train_labels
# selftest = matrix_test
# selftest_labels = test_labels
#
# X_train = selftrain
# X_test = selftest


# ##############################
data_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
# data_abnormal = np.loadtxt('./data.txt', dtype=str, delimiter=',')  # data: (200, 1601)

# data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.25t\\data.txt', dtype=str, delimiter=',')
# data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 250\\data.txt', dtype=str, delimiter=',')
# data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 zhengtai\\data.txt', dtype=str, delimiter=',')
data_abnormal = np.loadtxt('D:\\数据_刘嘉帅\\y_1 0.2sin(2pi-200)\\data.txt', dtype=str, delimiter=',')
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
silde_window = 400
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

X_train = selftrain
X_test = selftest






# # 生成一些示例数据（正常行为）
# np.random.seed(42)
# normal_data = np.random.normal(loc=0, scale=1, size=(100, 1))
#
# # 生成一些异常数据
# anomaly_data = np.random.normal(loc=5, scale=1, size=(20, 1))
#
# # 将数据合并
# data = np.vstack([normal_data, anomaly_data])
#
# # 创建一个具有两个状态的GaussianHMM模型
# model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
#
# # 将数据传递给模型进行训练
# model.fit(normal_data)
#
# # 预测每个数据点的概率
# probabilities = model.predict_proba(data)
#
# # 计算每个数据点属于异常状态的概率
# anomaly_probabilities = probabilities[:, 1]
#
# # 设置异常概率的阈值
# threshold = 0.5
#
# # 标记异常点
# anomalies = data[anomaly_probabilities > threshold]
#
# # 绘制结果
# plt.plot(data, label='Normal Data')
# plt.scatter(anomalies, anomalies, color='red', label='Anomalies')
# plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()







#############################################
#DeepSVDD
clf_DeepSVDD = DeepSVDD()
clf_DeepSVDD.fit(X_train, y=None)
scores_DeepSVDD = clf_DeepSVDD.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc_DeepSVDD, ap_DeepSVDD, f1_DeepSVDD = tabular_metrics(selftest_labels, scores_DeepSVDD)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"DeepSVDD - auc: {auc_DeepSVDD:.4f}, ap: {ap_DeepSVDD:.4f}, F1 Score: {f1_DeepSVDD:.4f}")
#zhengtai DeepSVDD - auc: 0.7673, ap: 0.8788, F1 Score: 0.7425
#############################################
#REPEN
clf_REPEN = REPEN()
clf_REPEN.fit(X_train, y=None)
scores_REPEN = clf_REPEN.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc_REPEN, ap_REPEN, f1_REPEN = tabular_metrics(selftest_labels, scores_REPEN)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"REPEN - auc: {auc_REPEN:.4f}, ap: {ap_REPEN:.4f}, F1 Score: {f1_REPEN:.4f}")
# REPEN - auc: 0.4602, ap: 0.6218, F1 Score: 0.7075
#############################################
#RDP
clf_RDP = RDP()
clf_RDP.fit(X_train, y=None)
scores_RDP = clf_RDP.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc_RDP, ap_RDP, f1_RDP = tabular_metrics(selftest_labels, scores_RDP)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"RDP - auc: {auc_RDP:.4f}, ap: {ap_RDP:.4f}, F1 Score: {f1_RDP:.4f}")
# RDP - auc: 0.9837, ap: 0.9912, F1 Score: 0.9625
#############################################
#RCA
clf = RCA()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"RCA - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# RCA - auc: 0.8714, ap: 0.9353, F1 Score: 0.8525
##############################################
#GOAD
clf = GOAD()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"GOAD - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# GOAD - auc: 0.8091, ap: 0.8949, F1 Score: 0.8225
##############################################
#NeuTraL
clf = NeuTraL()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"NeuTraL - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# NeuTraL - auc: 0.9956, ap: 0.9972, F1 Score: 0.9900
##############################################
#ICL
clf = ICL()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"ICL - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# ICL - auc: 0.9967, ap: 0.9974, F1 Score: 0.9975
##############################################
#DIF
clf = DeepIsolationForest()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"DeepIsolationForest - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# DeepIsolationForest - auc: 0.7122, ap: 0.8426, F1 Score: 0.7250
##############################################
#SLAD
clf = SLAD()
clf.fit(X_train, y=None)
scores = clf.decision_function(X_test)
# evaluation of tabular anomaly detection
from deepod.metrics import tabular_metrics
auc, ap, f1 = tabular_metrics(selftest_labels, scores)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"SLAD - auc: {auc:.4f}, ap: {ap:.4f}, F1 Score: {f1:.4f}")
# SLAD - auc: 0.9971, ap: 0.9982, F1 Score: 0.9925
##############################################
#
#
#
# ##############################################
# # TimesNet; time series anomaly detection methods
# clf = TimesNet()
# clf.fit(X_train)
# scores = clf.decision_function(X_test)
# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import point_adjustment # execute point adjustment for time series ad
# eval_metrics = ts_metrics(selftest_labels, scores)
# adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print("TimesNet","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
# ##############################################
# # AnomalyTransformer; time series anomaly detection methods
# clf = AnomalyTransformer()
# clf.fit(X_train)
# scores = clf.decision_function(X_test)
# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import point_adjustment # execute point adjustment for time series ad
# eval_metrics = ts_metrics(selftest_labels, scores)
# adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print("AnomalyTransformer","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)

##############################################
# # TranAD; time series anomaly detection methods
# clf = TranAD()
# clf.fit(X_train) #跑不通
# scores = clf.decision_function(X_test)
# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import point_adjustment # execute point adjustment for time series ad
# eval_metrics = ts_metrics(selftest_labels, scores)
# adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print("TranAD","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
##############################################

# COUTA; time series anomaly detection methods
clf = COUTA()
clf.fit(X_train)
scores = clf.decision_function(X_test)
# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment # execute point adjustment for time series ad
eval_metrics = ts_metrics(selftest_labels, scores)
adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("COUTA","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
# COUTA eval_metrics (0.8177, 0.9045625285519614, 0.8878973936005083, 0.7984031936127745, 1.0) adj_eval_metrics (1.0, 1.0, 0.9999950000249999, 1.0, 1.0)
##############################################

# # USAD; time series anomaly detection methods
# clf = USAD()
# clf.fit(X_train) #跑不通
# scores = clf.decision_function(X_test)
# # evaluation of time series anomaly detection
# from deepod.metrics import ts_metrics
# from deepod.metrics import point_adjustment # execute point adjustment for time series ad
# eval_metrics = ts_metrics(selftest_labels, scores)
# adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
# print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print("USAD	","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
# ##############################################

# DeepIsolationForestTS; time series anomaly detection methods
clf = DeepIsolationForestTS()
clf.fit(X_train)
scores = clf.decision_function(X_test)
# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment # execute point adjustment for time series ad
eval_metrics = ts_metrics(selftest_labels, scores)
adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("DeepIsolationForestTS","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
# DeepIsolationForestTS eval_metrics (0.9981375, 0.9991250162151453, 0.9899194335601385, 0.9974619289340102, 0.9825) adj_eval_metrics (1.0, 1.0, 0.9999950000249999, 1.0, 1.0)
##############################################
# DeepSVDDTS; time series anomaly detection methods
clf = DeepSVDDTS()
clf.fit(X_train)
scores = clf.decision_function(X_test)
# evaluation of time series anomaly detection
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment # execute point adjustment for time series ad
eval_metrics = ts_metrics(selftest_labels, scores)
adj_eval_metrics = ts_metrics(selftest_labels, point_adjustment(selftest_labels, scores))
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("DeepSVDDTS","eval_metrics",eval_metrics,"adj_eval_metrics",adj_eval_metrics)
# DeepSVDDTS eval_metrics (0.702775, 0.7247951358040138, 0.8680040038518196, 0.7854251012145749, 0.97) adj_eval_metrics (0.96, 0.9803921568627451, 0.9900940104163829, 0.9803921568627451, 1.0)

##############################################







