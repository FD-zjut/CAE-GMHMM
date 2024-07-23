def merge_files(file1, file2, file3, file4, output_file, max_lines_per_file=48000):
    files = [file1, file2, file3, file4]

    with open(output_file, 'w') as outfile:
        for file in files:
            line_count = 0
            with open(file, 'r') as infile:
                for line in infile:
                    if line_count < max_lines_per_file:
                        outfile.write(line)
                        line_count += 1
                    else:
                        break
# 使用示例
# file1 = 'D:/变工况数据集/实验用/中度点蚀故障/gear_pitting_M_speed_circulation_10Nm-1000rpm_10-20.txt'
# file2 = 'D:/变工况数据集/实验用/中度点蚀故障/gear_pitting_M_speed_circulation_10Nm-2000rpm_10-20.txt'
# file3 = 'D:/变工况数据集/实验用/中度点蚀故障/gear_pitting_M_speed_circulation_10Nm-1000rpm_25-35.txt'
# file4 = 'D:/变工况数据集/实验用/中度点蚀故障/gear_pitting_M_speed_circulation_10Nm-2000rpm_25-35.txt'

# file1 = 'D:/变工况数据集/实验用/中度断齿故障/teeth_break_M_speed_circulation_10Nm-1000rpm_10-20.txt'
# file2 = 'D:/变工况数据集/实验用/中度断齿故障/teeth_break_M_speed_circulation_10Nm-2000rpm_10-20.txt'
# file3 = 'D:/变工况数据集/实验用/中度断齿故障/teeth_break_M_speed_circulation_10Nm-1000rpm_25-35.txt'
# file4 = 'D:/变工况数据集/实验用/中度断齿故障/teeth_break_M_speed_circulation_10Nm-2000rpm_25-35.txt'

# file1 = 'D:/变工况数据集/实验用/中度裂纹故障/teeth_crack_M_speed_circulation_10Nm-1000rpm_10-20.txt'
# file2 = 'D:/变工况数据集/实验用/中度裂纹故障/teeth_crack_M_speed_circulation_10Nm-2000rpm_10-20.txt'
# file3 = 'D:/变工况数据集/实验用/中度裂纹故障/teeth_crack_M_speed_circulation_10Nm-1000rpm_25-35.txt'
# file4 = 'D:/变工况数据集/实验用/中度裂纹故障/teeth_crack_M_speed_circulation_10Nm-2000rpm_25-35.txt'

file1 = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-1000rpm_10-20.txt'
file2 = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-2000rpm_10-20.txt'
file3 = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-1000rpm_25-35.txt'
file4 = 'D:/变工况数据集/实验用/中度磨损故障/gear_wear_M_speed_circulation_10Nm-2000rpm_25-35.txt'

# file1 = 'D:/变工况数据集/实验用/正常样本/health_speed_circulation_10Nm-1000rpm_10-20.txt'
# file2 = 'D:/变工况数据集/实验用/正常样本/health_speed_circulation_10Nm-2000rpm_10-20.txt'
# file3 = 'D:/变工况数据集/实验用/正常样本/health_speed_circulation_10Nm-1000rpm_25-35.txt'
# file4 = 'D:/变工况数据集/实验用/正常样本/health_speed_circulation_10Nm-2000rpm_25-35.txt'



# output_file = 'D:/变工况数据集/实验用/异常样本/abnormal_pitting.txt'
# output_file = 'D:/变工况数据集/实验用/异常样本/abnormal_break.txt'
# output_file = 'D:/变工况数据集/实验用/异常样本/abnormal_crack.txt'
output_file = 'D:/变工况数据集/实验用/异常样本/abnormal_wear.txt'
# output_file = 'D:/变工况数据集/实验用/正常样本/normal_speed.txt'

merge_files(file1, file2, file3, file4, output_file)
