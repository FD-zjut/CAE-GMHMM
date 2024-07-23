# unsupervised methods
from deepod.models.tabular import RCA, DeepSVDD, REPEN, GOAD, NeuTraL, RDP, ICL, SLAD, DeepIsolationForest
from deepod.models.time_series import DeepIsolationForestTS, DeepSVDDTS, TranAD, USAD, COUTA, AnomalyTransformer, TimesNet
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
# from FC48_Adjust import AnomalyDetectionDataLoader



###KDDCUP
data = np.load('./kdd_cup.npz')
labels = data['kdd'][:, -1]  # 取最后一列数据
# print("labels:", labels.shape)#labels: (494021,)
features = data['kdd'][:, :-1]  # 取除最后一列外的所有列 #118维度(494021, 118)
# #cw
# data = np.loadtxt(data_path, delimiter=',')#data: (200, 1601)
# labels = data[:, -1]  # 取最后一列数据
# print("labels:", labels.shape)#labels: (200,)
# features = data[:,:-1]#取除最后一列外的所有列 #(200, 1600)

N, D = features.shape
# 0被标记为攻击，1为正常，但由于0多1少，则0最终被视为正常。
normal_data = features[labels == 1]
normal_labels = labels[labels == 1]

N_normal = normal_data.shape[0]

attack_data = features[labels == 0]
attack_labels = labels[labels == 0]

N_attack = attack_data.shape[0]
# 随机打乱攻击数据的索引
randIdx = np.arange(N_attack)
np.random.shuffle(randIdx)
N_train = N_attack // 2  # 将攻击数据的数量除以2，将一半的数据作为训练集。
# 划分训练集和测试集
selftrain = attack_data[randIdx[:N_train]]
selftrain_labels = attack_labels[randIdx[:N_train]]

selftest = attack_data[randIdx[N_train:]]
selftest_labels = attack_labels[randIdx[N_train:]]
# 将正常数据添加到测试集中
selftest = np.concatenate((selftest, normal_data), axis=0)
selftest_labels = np.concatenate((selftest_labels, normal_labels), axis=0)

X_train = selftrain
X_test = selftest
# self.test = self.test.reshape(-1, 9, 16)
# X_train = np.reshape(selftrain, (-1, 2400))#1200
# X_test = np.reshape(selftest, (-1, 2400))
# print(selftrain.shape)






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







