import pandas as pd

# # 读取数据
# data = pd.read_csv("FUDS-US06-DST-50-new—SOC2.csv")
#
# # 初始化SOC列
# data['SOC'] = 0.0
#
# # 初始容量和最大容量
# initial_capacity = data['Charge_Capacity(Ah)'].iloc[0]
# max_capacity = data['Charge_Capacity(Ah)'].max()
#
# # 初始化累计电荷量
# cumulative_charge = 0
#
# # 循环计算每一行的SOC
# for index, row in data.iterrows():
#     # 计算该行的电荷量变化
#     charge_change = row['Current(A)'] * row['Step_Time(s)'] / 3600  # 假设电流以安培为单位，时间以秒为单位，将时间单位转换为小时
#
#     # 更新累计电荷量
#     cumulative_charge += charge_change
#
#     # 计算该行的SOC
#     soc = ((initial_capacity + cumulative_charge) / max_capacity) * 100
#
#     # 将SOC值更新到对应行
#     data.at[index, 'SOC'] = soc
#
# # 打印结果
# print(data['SOC'])
# _______________________________________
excel_file = 'A1-007-DST-US06-FUDS-40-20120822.xlsx'
df =  pd.read_excel(excel_file,sheet_name=1,usecols=['Date_Time','Test_Time(s)','Step_Time(s)','Current(A)','Voltage(V)','Temperature (C)_1'])
df = df.iloc[17181:24764]

# 计算已充电容量
charge_capacity = (df['Current(A)'] * df['Step_Time(s)']).sum()

# 计算已放电容量
discharge_capacity = (abs(df['Current(A)']) * df['Step_Time(s)']).sum()

# 标称容量（假设为1.1Ah）
nominal_capacity = 1.1

# 计算SOC
soc = ((charge_capacity - discharge_capacity) / nominal_capacity) * 100

print(f"SOC: {soc:.2f}%")