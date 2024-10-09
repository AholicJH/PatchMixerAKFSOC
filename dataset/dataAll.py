import pandas as pd

# 读取CSV文件
file_path = 'DST_ALL_new.csv'  # 将'your_file.csv'替换为您实际的CSV文件路径
data = pd.read_csv(file_path)

# # 删除"Profile"列
# data = data.drop(columns=['Profile'])
#
# # 保存新的CSV文件（如果需要）
# new_file_path = 'DST_ALL_new.csv'  # 新CSV文件的路径和名称
# data.to_csv(new_file_path, index=False)



# 添加"Date"列，按秒递增
data.insert(0, 'Date', pd.date_range(start='2012-04-01 00:00:00', periods=len(data), freq='S'))

# 保存新的CSV文件（如果需要）
new_file_path = 'DST_ALL_new_date.csv'  # 新CSV文件的路径和名称
data.to_csv(new_file_path, index=False)
# 显示数据框的前几行
print(data.head())