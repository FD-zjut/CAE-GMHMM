import numpy as np

# 1. 打开.txt文件
txt_file_path1 = 'reshaped_data0.txt'
with open(txt_file_path1, 'r') as file:
    # 2. 读取文件内容
    data_NS0 = file.readlines()
txt_file_path2 = 'reshaped_data9.txt'
with open(txt_file_path2, 'r') as file:
    # 2. 读取文件内容
    data_GF9 = file.readlines()

# data_combine = np.concatenate((data_NS0, data_DF5, data_FF6, data_AF7, data_SPF8, data_GF9), axis=0)
data_combine = np.concatenate((data_NS0, data_GF9), axis=0)#先试试齿轮故障
# 3. 将数据保存为.npz文件
npz_file_path = 'pos_x_fault.npz'
np.savez(npz_file_path, data=data_combine)

# print(f'Data from {txt_file_path1} saved to {npz_file_path}')
