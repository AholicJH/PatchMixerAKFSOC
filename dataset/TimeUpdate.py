import pandas as pd

# 读取CSV文件
df = pd.read_csv('FUDS-US06-DST-10-true—SOC.csv')

# 将日期时间列转换为pandas的datetime格式
df['date'] = pd.to_datetime(df['date'])

# 确定开始时间和结束时间
start_time = df['date'].iloc[0]
end_time = df['date'].iloc[-1]
time_step = pd.Timedelta(seconds=1)

# 生成新的时间序列
new_time_series = pd.date_range(start=start_time, end=end_time, freq=time_step)

# 如果新的时间序列长度与原数据不匹配，选择短的那个
min_length = min(len(df), len(new_time_series))
df = df.iloc[:min_length]
new_time_series = new_time_series[:min_length]

# 更新date列
df['date'] = new_time_series.strftime('%Y-%m-%d %H:%M:%S')

# 保存更新后的DataFrame到CSV文件
df.to_csv('FUDS-US06-DST-10-true—SOC.csv', index=False)

