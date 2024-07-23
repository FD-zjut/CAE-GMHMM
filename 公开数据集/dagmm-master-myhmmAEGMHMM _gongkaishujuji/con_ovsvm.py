from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

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

data_nofault_pos = min_max_normalize_columns(data_nofault_pos, 0, 1)
spe_x1 = min_max_normalize_columns(spe_x1, 0, 1)
liju_x1 = min_max_normalize_columns(liju_x1, 0, 1)
data_GWF_pos = min_max_normalize_columns(data_GWF_pos, 0, 1)
spe_x2 = min_max_normalize_columns(spe_x2, 0, 1)
liju_x2 = min_max_normalize_columns(liju_x2, 0, 1)
# 再合并速度、位置、力矩
# 将数据重新形状为100行、1600列
reshaped_data_nofault_pos = np.reshape(data_nofault_pos, (400, 400))
reshaped_data_nofault_spe = np.reshape(spe_x1, (400, 400))
reshaped_data_nofault_liju = np.reshape(liju_x1, (400, 400))
reshaped_data_GWF_pos = np.reshape(data_GWF_pos, (400, 400))
reshaped_data_GWF_spe = np.reshape(spe_x2, (400, 400))
reshaped_data_GWF_liju = np.reshape(liju_x2, (400, 400))
# 将三个矩阵连接起来
matrix_nofault = np.concatenate(
    (reshaped_data_nofault_pos, reshaped_data_nofault_spe, reshaped_data_nofault_liju), axis=1)
matrix_data_GWF = np.concatenate((reshaped_data_GWF_pos, reshaped_data_GWF_spe, reshaped_data_GWF_liju), axis=1)

matrix_nofault = np.hstack((matrix_nofault, np.zeros((400, 1))))
matrix_data_GWF = np.hstack((matrix_data_GWF, np.ones((400, 1))))
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
selftrain = selftrain.reshape(-1, 3, 400)
selftrain_labels = attack_labels[randIdx[:N_train]]

selftest = attack_data[randIdx[N_train:]]
selftest_labels = attack_labels[randIdx[N_train:]]

selftest = np.concatenate((selftest, normal_data), axis=0)
selftest = selftest.reshape(-1, 3, 400)
selftest_labels = np.concatenate((selftest_labels, normal_labels), axis=0)

selftrain = selftrain.reshape(-1, 1200)
selftest = selftest.reshape(-1, 1200)

# 使用OCSVM
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1).fit(selftrain)

# 预测和评估
ocsvm_pred = ocsvm.predict(selftest)
ocsvm_pred = (ocsvm_pred == -1).astype(int)  # 将-1转换为异常(1)

# 计算指标
precision_ocsvm, recall_ocsvm, f_score_ocsvm, _ = prf(selftest_labels, ocsvm_pred, average='binary')
accuracy_ocsvm = accuracy_score(selftest_labels, ocsvm_pred)


print(f"OCSVM - Precision: {precision_ocsvm:.4f}, Recall: {recall_ocsvm:.4f}, F1 Score: {f_score_ocsvm:.4f}, Accuracy: {accuracy_ocsvm:.4f}")