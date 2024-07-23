import numpy as np


# def combine_files_chunk(file_paths, chunk_size, desired_length, output_file):
#     combined_data = []  # 用于存储组合后的数据块
#     a = 0
#     b = 0
#     file_positions = {file_path: 0 for file_path in file_paths}  # 用于跟踪每个文件读取到的位置
#
#     while a < 225:
#         for file_path in file_paths:
#             a += 1
#             b += 1
#             if a == 1:
#                 start_pos = file_positions[file_path]
#                 end_pos = start_pos + chunk_size
#
#             # 使用 anp.loadtxt 从文件中读取数据
#             file_data = np.loadtxt(file_path, dtype=str, delimiter="\n")
#
#             if b % 3 == 0:
#                 file_positions[file_path] = end_pos
#                 start_pos = file_positions[file_path]
#                 end_pos = start_pos + chunk_size
#
#             # 从文件数据中选取1600个数据并添加到数组中
#             combined_data.extend(file_data[start_pos:end_pos])
#
#             # 如果数组长度达到 desired_length，则退出循环
#             if len(combined_data) >= desired_length:
#                 combined_data = combined_data[:desired_length]  # 确保数据长度不超过 desired_length
#                 break
#
#         if len(combined_data) >= desired_length:
#             break
#
#     # 将合并的数据写入输出文件
#     with open(output_file, 'w') as f:
#         for line in combined_data:
#             f.write(f"{line}\n")


def combine_files_chunk(file_paths, chunk_size, desired_length, output_file):
    combined_data = []  # 用于存储组合后的数据块
    a = 0
    b = 0
    file_positions = {file_path: 0 for file_path in file_paths}  # 用于跟踪每个文件读取到的位置
    while a < 120:
        for file_path in file_paths:
            a += 1
            b += 1
            if a == 1:
                start_pos = file_positions[file_path]
                end_pos = start_pos + chunk_size
            with open(file_path, 'r') as file:
                # 从文件中读取数据
                if b % 3 == 0:
                    file_positions[file_path] = end_pos
                    start_pos = file_positions[file_path]
                    end_pos = start_pos + chunk_size
                file_data = file.readlines()
                # 从文件数据中选取1600个数据并添加到数组中
                combined_data.extend(file_data[start_pos:end_pos])

            # 如果数组长度达到360000，则退出循环
            if len(combined_data) >= desired_length:
                break


    with open(output_file, 'w') as f:
        for line in combined_data:
            f.write(f"{line}\n")


# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 无故障/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 无故障/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 无故障/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/r=10 y_1 0.993/data.txt', r'D:/数据_刘嘉帅/r=9.5 y_1 0.993/data.txt', r'D:/数据_刘嘉帅/r=9 y_1 0.993/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/y_1 0.25t/data.txt', r'D:/数据_刘嘉帅/r=9.5 y_1 0.25t/data.txt', r'D:/数据_刘嘉帅/r=9 y_1 0.25t/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/y_1 0.02sin(2pi-200)/data.txt', r'D:/数据_刘嘉帅/r=9.5 y_1 0.02sin(2pi-200)/data.txt', r'D:/数据_刘嘉帅/r=9 y_1 0.02sin(2pi-200)/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/y_1 zhengtai/data.txt', r'D:/数据_刘嘉帅/r=9.5 y_1 zhengtai/data.txt', r'D:/数据_刘嘉帅/r=9 y_1 zhengtai/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 DDA l=120  t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 DDA l=120  t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 DDA l=120  t 1200/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 GWA l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 GWA l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 GWA l=120 t 1200/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 FAB l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 FAB l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 FAB l=120 t 1200/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 ADA l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 ADA l=120 t 1200/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 ADA l=120 t 1200/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 DDA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 DDA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 DDA l=120 t 800/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 GWA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 GWA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 GWA l=120 t 800/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 FAB l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 FAB l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 FAB l=120 t 800/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 ADA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 ADA l=120 t 800/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 ADA l=120 t 800/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 DDA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 DDA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 DDA l=120 t 400/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 GWA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 GWA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 GWA l=120 t 400/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 FBA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 FBA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 FBA l=120 t 400/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/9 ADA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5 ADA l=120 t 400/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/10 ADA l=120 t 400/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/r=10 无故障/data.txt', r'D:/数据_刘嘉帅/r=9.5 无故障/data.txt', r'D:/数据_刘嘉帅/r=9 无故障/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.5.18敏感性/10/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9.5/data.txt', r'D:/数据_刘嘉帅/24.5.18敏感性/9/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 GWA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 GWA 0 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 GWA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 DDA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 DDA 0 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 DDA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 FBA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 FBA 0 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 FBA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 ADA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 ADA 0 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 ADA 0 30/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 ADA 400 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 ADA 0 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 ADA 0 45/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 FBA 400 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 FBA 0 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 FBA 0 45/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 GWA 400 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 GWA 0 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 GWA 0 45/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 DDA 400 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 DDA 0 45/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 DDA 0 45/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 ADA 400 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 ADA 0 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 ADA 0 60/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 FBA 400 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 FBA 0 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 FBA 0 60/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 GWA 400 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 GWA 0 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 GWA 0 60/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 DDA 400 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 DDA 0 60/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 DDA 0 60/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 ADA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 ADA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 ADA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 FBA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 FBA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 FBA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 GWA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 GWA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 GWA 0 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 DDA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 DDA 400 30/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 DDA 0 30/data.txt']

# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 ADA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 ADA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 ADA 400 30/data.txt']
file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 FBA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 FBA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 FBA 400 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 GWA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 GWA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 GWA 400 30/data.txt']
# file_paths = [r'D:/数据_刘嘉帅/24.6.4敏感性/10 DDA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9.5 DDA 0 30 31/data.txt', r'D:/数据_刘嘉帅/24.6.4敏感性/9 DDA 400 30/data.txt']

chunk_size = 1600
desired_length = 480000
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data.txt'
# output_file = r'D:\数据_刘嘉帅\combined_data_abnormal_0.993.txt'
# output_file = r'D:\数据_刘嘉帅\combined_data_abnormal_0.25t.txt'
# output_file = r'D:\数据_刘嘉帅\combined_data_abnormal_sin.txt'
# output_file = r'D:\数据_刘嘉帅\combined_data_abnormal_zhengtai.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t1200_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t1200_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t1200_FAB.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t1200_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t800_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t800_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t800_FAB.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t800_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t400_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t400_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t400_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_abnormal_t400_FAB.txt'

# output_file = r'D:\数据_刘嘉帅\24.5.18敏感性\combined_data_normal_5.31.txt'

# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_30_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_30_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_30_FBA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_30_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_45_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_45_FBA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_45_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_45_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_60_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_60_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_60_FBA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_400_60_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_FBA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_ADA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_FBA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_2000_30_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_3600_30_DDA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_3600_30_GWA.txt'
# output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_3600_30_ADA.txt'
output_file = r'D:\数据_刘嘉帅\24.6.4敏感性\combined_data_abnormal_3600_30_FBA.txt'

combine_files_chunk(file_paths, chunk_size, desired_length, output_file)
