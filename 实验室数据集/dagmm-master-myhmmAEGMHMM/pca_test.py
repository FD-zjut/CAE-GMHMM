import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

# test_labels = np.vstack((np.zeros((600, 1)), np.ones((1200, 1))))
# test_labels = np.vstack((np.zeros((200, 1)),np.ones((200, 1)),2*np.ones((200, 1)), 3*np.ones((400, 1)),4*np.ones((400, 1)),5*np.ones((400, 1))))
test_labels = np.vstack((np.zeros((200, 1)),np.ones((200, 1)),2*np.ones((200, 1)), 3*np.ones((100, 1)),4*np.ones((100, 1)),5*np.ones((100, 1))))
# test_labels = np.vstack((np.zeros((80, 1)),np.ones((80, 1)),2*np.ones((80, 1)), 3*np.ones((20, 1)),4*np.ones((20, 1)),5*np.ones((20, 1))))


test_z = np.loadtxt('./test_z.txt', dtype=str, delimiter=' ').astype('float32')
# row_to_delete = np.concatenate([np.arange(700,1000),np.arange(1100,1400),np.arange(1500,1800)])
row_to_delete = np.concatenate([np.arange(600,900),np.arange(1100,1400),np.arange(1500,1800)])
test_z = np.delete(test_z,row_to_delete,axis=0)


test_z_size = test_z.shape[0]
# N_attack = matrix_nofault1.shape[0]
randIdx = np.arange(test_z_size)
# print("randIdx", randIdx, randIdx.shape)
np.random.shuffle(randIdx)
test_z = test_z[randIdx]
test_labels = test_labels[randIdx]



# test_z = np.loadtxt('./matrix_test.txt', dtype=str, delimiter=' ').astype('float32')


# minmax = MinMaxScaler()
# scaled_x = minmax.fit_transform(test_z)
# pca = PCA(n_components=2)
# pca_x = pca.fit_transform(scaled_x)
# print(test_z.shape)
# print(scaled_x.shape)
# print(pca_x.shape)
# # plt.scatter(pca_x[np.squeeze(test_labels ==0), 0], pca_x[np.squeeze(test_labels ==0), 1], label='label0', c = 'blue', marker='^')
# # plt.scatter(pca_x[np.squeeze(test_labels ==1), 0], pca_x[np.squeeze(test_labels ==1), 1], label='label1', c = 'red', marker='^')
#
#
# plt.scatter(pca_x[np.squeeze(test_labels ==0), 0], pca_x[np.squeeze(test_labels ==0), 1], label='label0', c = 'blue', marker='^')
# plt.scatter(pca_x[np.squeeze(test_labels ==1), 0], pca_x[np.squeeze(test_labels ==1), 1], label='label0', c = 'navy', marker='^')
# plt.scatter(pca_x[np.squeeze(test_labels ==2), 0], pca_x[np.squeeze(test_labels ==2), 1], label='label0', c = 'cornflowerblue', marker='^')
# plt.scatter(pca_x[np.squeeze(test_labels ==3), 0], pca_x[np.squeeze(test_labels ==3), 1], label='label1', c = 'red', marker='^')
# plt.scatter(pca_x[np.squeeze(test_labels ==4), 0], pca_x[np.squeeze(test_labels ==4), 1], label='label1', c = 'tomato', marker='^')
# plt.scatter(pca_x[np.squeeze(test_labels ==5), 0], pca_x[np.squeeze(test_labels ==5), 1], label='label1', c = 'sandybrown', marker='^')
#
# plt.legend()
# plt.title('PCA Example')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()
# print(pca_x.shape)



# # t-SNE
np.random.seed(0)
tsne = TSNE(n_components=2,perplexity = 20, n_iter=10000, init='random', random_state=0, learning_rate= 5)
# tsne = TSNE(n_components=2,random_state=0)
embedded_data = tsne.fit_transform(test_z)

# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# embedded_data = tsne.fit_transform(test_z)



# '''嵌入空间可视化'''
# x_min, x_max = embedded_data.min(0), embedded_data.max(0)
# X_norm = (embedded_data - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()



##二维
fig = plt.figure(figsize=(8, 8))
x_min, x_max = embedded_data.min(0), embedded_data.max(0)
embedded_data = (embedded_data - x_min) / (x_max - x_min)  # 归一化
# plt.scatter(embedded_data[:,0],embedded_data[:,1])
plt.scatter(embedded_data[np.squeeze(test_labels ==0), 0], embedded_data[np.squeeze(test_labels ==0), 1], label='label0', c = 'blue', marker='o')
plt.scatter(embedded_data[np.squeeze(test_labels ==1), 0], embedded_data[np.squeeze(test_labels ==1), 1], label='label0', c = 'navy', marker='o')
plt.scatter(embedded_data[np.squeeze(test_labels ==2), 0], embedded_data[np.squeeze(test_labels ==2), 1], label='label0', c = 'cornflowerblue', marker='o')
plt.scatter(embedded_data[np.squeeze(test_labels ==3), 0], embedded_data[np.squeeze(test_labels ==3), 1], label='label1', c = 'red', marker='o')
plt.scatter(embedded_data[np.squeeze(test_labels ==4), 0], embedded_data[np.squeeze(test_labels ==4), 1], label='label1', c = 'tomato', marker='o')
plt.scatter(embedded_data[np.squeeze(test_labels ==5), 0], embedded_data[np.squeeze(test_labels ==5), 1], label='label1', c = 'sandybrown', marker='o')

plt.xlabel('t-SNE 1',size=20)
plt.ylabel('t-SNE 2',size=20)
plt.xticks(size=20)
plt.yticks(size=20)

plt.savefig('test_z.png',dpi=1000,bbox_inches = 'tight')  # 保存为名为"output.png"的PNG文件
plt.show()


# #三维
# # %matplotlib notebook
# fig = plt.figure(figsize=(10,8))
# # fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.view_init(elev=45.,azim=60.)
#
# ax.scatter(embedded_data[np.squeeze(test_labels ==0), 0], embedded_data[np.squeeze(test_labels ==0), 1],embedded_data[np.squeeze(test_labels ==0), 2], label='label0', c = 'blue', marker='^')
# ax.scatter(embedded_data[np.squeeze(test_labels ==1), 0], embedded_data[np.squeeze(test_labels ==1), 1],embedded_data[np.squeeze(test_labels ==1), 2], label='label0', c = 'navy', marker='^')
# ax.scatter(embedded_data[np.squeeze(test_labels ==2), 0], embedded_data[np.squeeze(test_labels ==2), 1],embedded_data[np.squeeze(test_labels ==2), 2], label='label0', c = 'cornflowerblue', marker='^')
# ax.scatter(embedded_data[np.squeeze(test_labels ==3), 0], embedded_data[np.squeeze(test_labels ==3), 1],embedded_data[np.squeeze(test_labels ==3), 2], label='label1', c = 'red', marker='^')
# ax.scatter(embedded_data[np.squeeze(test_labels ==4), 0], embedded_data[np.squeeze(test_labels ==4), 1],embedded_data[np.squeeze(test_labels ==4), 2], label='label1', c = 'tomato', marker='^')
# ax.scatter(embedded_data[np.squeeze(test_labels ==5), 0], embedded_data[np.squeeze(test_labels ==5), 1],embedded_data[np.squeeze(test_labels ==5), 2], label='label1', c = 'sandybrown', marker='^')
#
# ax.set_xlabel('dimension 1')
# ax.set_ylabel('dimension 2')
# ax.set_zlabel('dimension 3')
# plt.tight_layout()
# plt.show()