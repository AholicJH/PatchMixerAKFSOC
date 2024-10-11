import math

import pandas as pd
from matplotlib import pyplot as plt

# 读取Excel文件
excel_file = 'A1-007-DST-US06-FUDS-25-20120827.xlsx'
data =  pd.read_excel(excel_file,sheet_name=1,usecols=['Date_Time','Test_Time(s)','Current(A)','Voltage(V)','Charge_Capacity(Ah)','Temperature (C)_1'])
# print(df)
# print(df.head())
# 将数据保存为CSV文件
# csv_file = 'DST-US06-FUDS-25.csv'
# df.to_csv(csv_file, index=False)
#
# print(f'{excel_file} 已成功转换为 {csv_file}')
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取Excel文件
# data = pd.read_excel('DST-US06-FUDS-25.xlsx',sheet_name=1)
#
# # 选择特定行范围（从第956行到第8369行）
data = data.iloc[980:8367]

# 提取Voltage(V)和Test_Time(s)列
voltage = data['Current(A)']
test_time = data['Test_Time(s)']
# 将x的值减去5000
test_time -= 5000
# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(test_time, voltage, marker='o', linestyle='-')

#设置图表标题和坐标轴标签
plt.title('Current(A)')
plt.xlabel('Test Time(s)')
plt.ylabel('Current(A)')

# 设置y轴范围和刻度间隔
plt.yticks([i for i in range(-4, 3)])

# 显示图例
plt.legend(['Current(A)'])
# 显示图表
plt.grid(True)
plt.show()
# import pandas as pd
#
# # 读取Excel文件
# data = pd.read_excel('A1-007-DST-US06-FUDS-25-20120827.xlsx', sheet_name="Channel_1-006",usecols=['Date_Time','Test_Time(s)','Current(A)','Voltage(V)','Charge_Capacity(Ah)','Temperature (C)_1'])
#
# # 选择特定行范围（从第979行到第8369行）
# df = data.iloc[17068:24470]  # iloc索引从0开始，所以第979行对应索引978
#
# # 计算 SOC
# df['SOC'] = ( df['Current(A)'] * df['Test_Time(s)'] / 3600 / df['Charge_Capacity(Ah)']) * 100
# # 保存到新的CSV文件中
# df.to_csv('FUDS-US06-DST-25-SOC.csv', index=False)
