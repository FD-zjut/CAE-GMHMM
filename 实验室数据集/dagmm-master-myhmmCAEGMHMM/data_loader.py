import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
# from FC48_Adjust import AnomalyDetectionDataLoader
# def combine_files_chunk(file_paths, chunk_size, desired_length, output_file):
#     combined_data = []  # 用于存储组合后的数据块
#     a = 0
#     b = 0
#     file_positions = {file_path: 0 for file_path in file_paths}  # 用于跟踪每个文件读取到的位置
#     while a < 225:
#         for file_path in file_paths:
#             a += 1
#             b += 1
#             if a == 1:
#                 start_pos = file_positions[file_path]
#                 end_pos = start_pos + chunk_size
#             with open(file_path, 'r') as file:
#                 # 从文件中读取数据
#                 if b % 3 == 0:
#                     file_positions[file_path] = end_pos
#                     start_pos = file_positions[file_path]
#                     end_pos = start_pos + chunk_size
#                 file_data = file.readlines()
#                 # 从文件数据中选取1600个数据并添加到数组中
#                 combined_data.extend(file_data[start_pos:end_pos])
#
#             # 如果数组长度达到360000，则退出循环
#             if len(combined_data) >= desired_length:
#                 break
#
#
#     with open(output_file, 'w') as f:
#         for line in combined_data:
#             f.write(f"{line}\n")


# 示例用法
# file_paths = [r'D:\数据_刘嘉帅\r=9 无故障\data.txt', r'D:\数据_刘嘉帅\r=9.5 无故障\data.txt', r'D:\数据_刘嘉帅\r=10 无故障\data.txt']
# chunk_size = 1600
#
# combined_data = combine_files_chunk(file_paths, chunk_size)

class KDD99Loader(object):
    def __init__(self, data_path, mode="train"):
        self.mode=mode
        #正常数据
        # data_r10_normal = np.loadtxt('./data_nofault.txt', dtype=str, delimiter=',')
        # data_r10_normal = np.loadtxt('D:/数据_刘嘉帅/r=10 无故障/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_normal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 无故障/data.txt', dtype=str, delimiter=',')
        # data_r5_normal = np.loadtxt('D:/数据_刘嘉帅/r=9 无故障/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt(data_path, dtype=str, delimiter=',')
        # data_pin_normal = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data.txt', dtype=str, delimiter=',')
        # data_pin_abnormal = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_normal_5.31.txt', dtype=str, delimiter=',')

        data_pin_normal = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_normal_5.31.txt', dtype=str, delimiter=',')
        # data_pin_abnormal = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data.txt', dtype=str, delimiter=',')


#敏感性5.18
        # data_pin_abnormal_933 = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_0.993.txt', dtype=str, delimiter=',')
        # data_pin_abnormal_25t = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_0.25t.txt', dtype=str, delimiter=',')
        # data_pin_abnormal_sin = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_sin.txt', dtype=str, delimiter=',')
        # data_pin_abnormal_zhengtai = np.loadtxt('D:/数据_刘嘉帅/combined_data_abnormal_zhengtai.txt', dtype=str, delimiter=',')
        #
        # data_pin_abnormal_DDA_t1200 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t1200_DDA.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_GWA_t1200 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t1200_GWA.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_FAB_t1200 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t1200_FAB.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_ADA_t1200 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t1200_ADA.txt', dtype=str,delimiter=',')
        #
        # data_pin_abnormal_ADA_t800 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t800_ADA.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_GWA_t800 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t800_GWA.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_FAB_t800 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t800_FAB.txt', dtype=str,delimiter=',')
        # data_pin_abnormal_DDA_t800 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t800_DDA.txt', dtype=str,delimiter=',')
        #
        # data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t400_ADA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t400_GWA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t400_FAB.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.5.18敏感性/combined_data_abnormal_t400_DDA.txt', dtype=str,
        #                                         delimiter=',')
#************************

#敏感性6.4
        # data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_30_GWA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_30_DDA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_30_FBA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_30_ADA.txt', dtype=str,
        #                                         delimiter=',')

        # data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_45_ADA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_45_FBA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_45_GWA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_45_DDA.txt', dtype=str,
        #                                         delimiter=',')

        # data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_60_ADA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_60_FBA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_60_GWA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_400_60_DDA.txt', dtype=str,
        #                                         delimiter=',')

        # data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_2000_30_ADA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_2000_30_FBA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_2000_30_GWA.txt', dtype=str,
        #                                         delimiter=',')
        # data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_2000_30_DDA.txt', dtype=str,
        #                                         delimiter=',')

        data_pin_abnormal_ADA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_3600_30_ADA.txt', dtype=str,
                                                delimiter=',')
        data_pin_abnormal_FAB_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_3600_30_FBA.txt', dtype=str,
                                                delimiter=',')
        data_pin_abnormal_GWA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_3600_30_GWA.txt', dtype=str,
                                                delimiter=',')
        data_pin_abnormal_DDA_t400 = np.loadtxt('D:/数据_刘嘉帅/24.6.4敏感性/combined_data_abnormal_3600_30_DDA.txt', dtype=str,
                                                delimiter=',')

        #半径为10的各种异常数据
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 0.25t/data.txt', dtype=str, delimiter=',')  # DDA 0.1
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 250/data.txt', dtype=str, delimiter=',')  # FBA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 zhengtai/data.txt', dtype=str, delimiter=',')  # ADA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')#GWA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 25/data.txt', dtype=str, delimiter=',')#FBA 25ok
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 0.002sin(2pi-400)/data.txt', dtype=str, delimiter=',')  # GWA
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 300/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 42sin(2pi -200)/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 (start, stop, length)/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 zhengtai/data.txt', dtype=str, delimiter=',')
        # data_r10_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=10 y_1 0.993/data.txt', dtype=str, delimiter=',')

        # 半径为9.5的各种异常数据
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 0.25t/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 250/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 zhengtai/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 0.993/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 42sin(2pi -200)/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 300/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 (start, stop, length)/data.txt', dtype=str, delimiter=',')
        # data_r7dot5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9.5 y_1 zhengtai/data.txt', dtype=str, delimiter=',')

        # 半径为19的各种异常数据
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.25t/data.txt', dtype=str, delimiter=',')  # DDA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 250/data.txt', dtype=str, delimiter=',')  # FBA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 zhengtai/data.txt', dtype=str, delimiter=',')#ADA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.02sin(2pi-200)/data.txt', dtype=str, delimiter=',')#GWA
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 42sin(2pi -200)/data.txt', dtype=str, delimiter=',')
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 300/data.txt', dtype=str, delimiter=',')
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 (start, stop, length)/data.txt', dtype=str, delimiter=',')
        # data_r5_abnormal = np.loadtxt('D:/数据_刘嘉帅/r=9 y_1 0.993/data.txt', dtype=str, delimiter=',')

        #删除数据中不需要的
        # data_r10_normal = np.delete(data_r10_normal, np.array([0, 5, 8]), 1)
        # data_r7dot5_normal = np.delete(data_r7dot5_normal, np.array([0, 5, 8]), 1)
        # data_r5_normal = np.delete(data_r5_normal, np.array([0, 5, 8]), 1)
        # data_r10_abnormal = np.delete(data_r10_abnormal, np.array([0, 5, 8]), 1)
        # data_r7dot5_abnormal = np.delete(data_r7dot5_abnormal, np.array([0, 5, 8]), 1)
        # data_r5_abnormal = np.delete(data_r5_abnormal, np.array([0, 5, 8]), 1)


        data_pin_normal = np.delete(data_pin_normal, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_933 = np.delete(data_pin_abnormal_933, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_25t = np.delete(data_pin_abnormal_25t, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_sin = np.delete(data_pin_abnormal_sin, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_zhengtai = np.delete(data_pin_abnormal_zhengtai, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_DDA_t1200 = np.delete(data_pin_abnormal_DDA_t1200, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_GWA_t1200 = np.delete(data_pin_abnormal_GWA_t1200, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_FAB_t1200 =  np.delete(data_pin_abnormal_FAB_t1200, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_ADA_t1200 = np.delete(data_pin_abnormal_ADA_t1200, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_ADA_t800 = np.delete(data_pin_abnormal_ADA_t800, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_GWA_t800 = np.delete(data_pin_abnormal_GWA_t800, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_FAB_t800 = np.delete(data_pin_abnormal_FAB_t800, np.array([0, 5, 8]), 1)
        # data_pin_abnormal_DDA_t800 = np.delete(data_pin_abnormal_DDA_t800, np.array([0, 5, 8]), 1)
        data_pin_abnormal_ADA_t400 = np.delete(data_pin_abnormal_ADA_t400, np.array([0, 5, 8]), 1)
        data_pin_abnormal_GWA_t400 = np.delete(data_pin_abnormal_GWA_t400, np.array([0, 5, 8]), 1)
        data_pin_abnormal_FAB_t400 = np.delete(data_pin_abnormal_FAB_t400, np.array([0, 5, 8]), 1)
        data_pin_abnormal_DDA_t400 = np.delete(data_pin_abnormal_DDA_t400, np.array([0, 5, 8]), 1)
        # data_pin_abnormal = np.delete(data_pin_abnormal, np.array([0, 5, 8]), 1)
        #将不同半径即工况下的数据进行合并（包括异常与异常间的合并，正常和正常间的合并）
        _all_length = 160000
        _all_length_abnormal = 64000
        _all_length_pin_40 = 192000
        _all_length_pin_30 = 144000
        _all_length_pin_20 = 96000
        # matrix_nofault = np.concatenate((data_r10_normal[:_all_length, :],
        #                                  data_r7dot5_normal[:_all_length, :],
        #                                  data_r5_normal[:_all_length, :]), axis=0)
        # matrix_addfault = np.concatenate((data_r10_abnormal[:_all_length_abnormal, :],
        #                                   data_r7dot5_abnormal[:_all_length_abnormal, :],
        #                                   data_r5_abnormal[:_all_length_abnormal, :]), axis=0)
        matrix_nofault = data_pin_normal
        # matrix_addfault = data_pin_abnormal_933
        # matrix_addfault = data_pin_abnormal_25t
        # matrix_addfault = data_pin_abnormal_sin
        # matrix_addfault = data_pin_abnormal_zhengtai
        # matrix_addfault = data_pin_abnormal_DDA_t1200
        # matrix_addfault = data_pin_abnormal_GWA_t1200
        # matrix_addfault = data_pin_abnormal_FAB_t1200
        # matrix_addfault = data_pin_abnormal_ADA_t1200
        # matrix_addfault = data_pin_abnormal_ADA_t800
        # matrix_addfault = data_pin_abnormal_GWA_t800
        # matrix_addfault = data_pin_abnormal_FAB_t800
        # matrix_addfault = data_pin_abnormal_DDA_t800
        matrix_addfault = data_pin_abnormal_ADA_t400
        # matrix_addfault = data_pin_abnormal_GWA_t400
        # matrix_addfault = data_pin_abnormal_FAB_t400
        # matrix_addfault = data_pin_abnormal_DDA_t400
        # data_pin_abnormal = data_pin_abnormal[:192000, :]
        # matrix_addfault = data_pin_abnormal

        # matrix_addfault = data_pin_abnormal_DDA_t1200[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_GWA_t1200[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_FAB_t1200[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_ADA_t1200[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_ADA_t800[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_GWA_t800[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_FAB_t800[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_DDA_t800[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_ADA_t400[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_GWA_t400[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_FAB_t400[:_all_length_pin_30,:]
        # matrix_addfault = data_pin_abnormal_DDA_t400[:_all_length_pin_30,:]

        # matrix_addfault = data_pin_abnormal_DDA_t1200[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_GWA_t1200[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_FAB_t1200[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_ADA_t1200[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_ADA_t800[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_GWA_t800[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_FAB_t800[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_DDA_t800[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_ADA_t400[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_GWA_t400[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_FAB_t400[:_all_length_pin_20,:]
        # matrix_addfault = data_pin_abnormal_DDA_t400[:_all_length_pin_20,:]


        # 归一化
        def min_max_normalize_columns(data_column, min_val, max_val):
            data_column = data_column.astype(float)  # Convert data to float type
            return (data_column - np.min(data_column)) / (np.max(data_column) - np.min(data_column)) * (
                    max_val - min_val) + min_val


        aaaa = _all_length // 2
        slid_window =2400
        #进行了列方向的最小-最大归一化处理，将所有列的值缩放到 [0, 1] 的范围内
        matrix_nofault = min_max_normalize_columns(matrix_nofault, 0, 1)
        matrix_addfault = min_max_normalize_columns(matrix_addfault, 0, 1)
        #重新塑造为一个形状为 (3, _all_length, -1) 的数组
        matrix_nofault1 = np.reshape(matrix_nofault, (3, _all_length, -1))
        # 选取matrix_nofault1的一半作为训练集
        matrix_nofault2_train = matrix_nofault1[:, :_all_length // 2, :]  # (3,80000,6)
        matrix_nofault2_test = matrix_nofault1[:, _all_length // 2:, :]
        matrix_nofault3_train = np.reshape(matrix_nofault2_train, (240000, -1))
        matrix_nofault3_test = np.reshape(matrix_nofault2_test, (240000, -1))
        #进行划窗
        matrix_nofault4_train = np.reshape(matrix_nofault3_train, (slid_window, 240000 // slid_window, -1))
        matrix_nofault4_test = np.reshape(matrix_nofault3_test, (slid_window, 240000 // slid_window, -1))
        #重新排列数组的维度顺序
        matrix_nofault5_train = np.transpose(matrix_nofault4_train, (1, 2, 0))  # (600,6,400) (150,6,1600)
        matrix_nofault5_test = np.transpose(matrix_nofault4_test, (1, 2, 0))  #

	# dataset1 = matrix_nofault5_train[:200]
	# dataset2 = matrix_nofault5_train[200:400]
	# dataset3 = matrix_nofault5_train[400:]
	# np.random.shuffle(dataset1)
	# np.random.shuffle(dataset2)
	# np.random.shuffle(dataset3)
	# matrix_nofault5_train = np.concatenate((dataset1, dataset2, dataset3), axis=0)

        # matrix_addfault1 = np.reshape(matrix_addfault, (slid_window, _all_length_pin_40 // slid_window, -1))
        # matrix_addfault2 = np.transpose(matrix_addfault1, (1, 2, 0))
        matrix_addnofault1 = matrix_addfault[:144000, :]
        matrix_addnofault1 = np.reshape(matrix_addnofault1, (slid_window, 144000 // slid_window, -1))
        matrix_addnofault2 = np.transpose(matrix_addnofault1, (1, 2, 0))

        matrix_addfault3 = matrix_addfault[144000:, :]
        matrix_addfault3 = np.reshape(matrix_addfault3, (slid_window, 48000 // slid_window, -1))
        matrix_addfault4 = np.transpose(matrix_addfault3, (1, 2, 0))
        #形成测试集
        # matrix_test = np.vstack((matrix_nofault5_test, matrix_addnofault2, matrix_addfault4))
        matrix_test = np.vstack((matrix_nofault5_test, matrix_addfault4))
        #形成测试集和训练集的标签
        train_labels = np.vstack((np.zeros((100, 1))))
        # test_labels = np.vstack((np.zeros((600, 1)), np.ones((480, 1)))
        # test_labels = np.vstack((np.zeros((240, 1)), np.ones((30, 1))))
        test_labels = np.vstack((np.zeros((100, 1)), np.ones((20, 1))))
        # test_labels = np.vstack((np.zeros((300, 1)), np.zeros((240, 1))))

        self.train = matrix_nofault5_train
        # self.train = self.train.reshape(-1, 2400)
        self.train_labels = train_labels

        self.test = matrix_test
        self.test_labels = test_labels
        # self.test = self.test.reshape(-1, 2400)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    dataset = KDD99Loader(data_path, mode)

    shuffle = False #false是正常不打乱，而true是打乱顺序
    #cw
    # if mode == 'train':
    #     shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
