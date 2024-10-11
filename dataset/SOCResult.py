import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# # 读取 Excel 文件
excel_file = 'A1-007-DST-US06-FUDS-10-20120815.xlsx'
df =  pd.read_excel(excel_file,sheet_name=1,usecols=['Date_Time','Test_Time(s)','Current(A)','Voltage(V)','Temperature (C)_1'])
df = df.iloc[16804:23879]
# 将Date_Time列转换为datetime格式
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df['Current(A)']= -df['Current(A)']
#861到8344 ----859到8343
# 计算 SOC 参考值
Qn = 1.1 * 3600  # 电池额定容量，单位为安时（Ah）1100mah
SOC_real = [1]  # 初始 SOC 值，假设为 100%

for i in range(1, len(df)):
    # 使用安时积分法计算 SOC 参考值
    SOC = SOC_real[-1] - (df['Current(A)'].iloc[i-1] * (df['Test_Time(s)'].iloc[i] - df['Test_Time(s)'].iloc[i-1])) / Qn
    SOC_real.append(SOC)
# 删除指定的列
columns_to_drop = ['Test_Time(s)']
df = df.drop(columns=columns_to_drop)
df['SOC(%)'] = SOC_real
df.to_csv('FUDS-US06-DST-10-true—SOC.csv', index=False,date_format='%Y-%m-%d %H:%M:%S')

# ___________________________________________________________

