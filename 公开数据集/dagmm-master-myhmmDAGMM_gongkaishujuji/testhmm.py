import numpy as np
from hmmlearn import hmm
# 准备训练数据和测试数据


data = np.loadtxt('D:/cw/dagmm-master-myhmm1127/data_no_gwf.txt', delimiter=',')
labels = data[:, -1]  # 取最后一列数据
# print("labels:", labels.shape, labels)  # labels: (200,)
features = data[:, :-1]  # 取除最后一列外的所有列 #(200, 1600)

normal_data = features[labels == 0]
normal_labels = labels[labels == 0]

N_normal = normal_data.shape[0]

attack_data = features[labels == 1]
attack_labels = labels[labels == 1]

N_attack = attack_data.shape[0]
randIdx = np.arange(N_normal)
np.random.shuffle(randIdx)
N_train = N_normal // 2

train_data = normal_data[randIdx[:N_train]]
train_data_labels = normal_labels[randIdx[:N_train]]

test_data = normal_data[randIdx[N_train:]]
test_data_labels = normal_labels[randIdx[N_train:]]

test_data = np.concatenate((test_data, attack_data),axis=0)
test_data_labels = np.concatenate((test_data_labels, attack_labels),axis=0)


num_states = 3
model = hmm.GaussianHMM(n_components=num_states, covariance_type="full", n_iter=100)
model.fit(train_data)

log_likelihood = model.score(test_data)
# print("log_likelihood",log_likelihood)
posterior_prob = np.exp(model.score_samples(test_data)[1])
print("posterior_prob",posterior_prob)
threshold = np.percentile(np.exp(model.score_samples(train_data)[1]), 95)
print(threshold,"threshold")
is_anomaly = posterior_prob < threshold
print(is_anomaly,"is_anomaly")
# threshold = np.percentile(log_likelihood, 95)
print(threshold,"threshold")
