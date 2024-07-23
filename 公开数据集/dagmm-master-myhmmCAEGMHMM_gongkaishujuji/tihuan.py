import pandas as pd

# 读取CSV文件
csv_file = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-1000rpm.csv'  # 替换为你的CSV文件名
# csv_file = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-2000rpm.csv'

df = pd.read_csv(csv_file)

# 将DataFrame写入TXT文件
txt_file = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-1000rpm.txt'
# txt_file = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-2000rpm.txt'

df.to_csv(txt_file, sep=',', index=False)
