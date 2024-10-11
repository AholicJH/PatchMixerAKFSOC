import numpy as np
import matplotlib.pyplot as plt

setting = 'loss_flag2_lr0.0001_dm512_test_PatchMixer_A123Custom_ftMS_sl48_pl26_p16s8_random2021_0'
# 定义文件夹路径
folder_path = './' + setting + '/'
# 加载数据
preds_flat = np.load(folder_path + 'preds_flat.npy')
trues_flat = np.load(folder_path + 'trues_flat.npy')

print('preds_flat:', preds_flat.shape)
print('trues_flat:', trues_flat.shape)

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(preds_flat, label='Predictions', linewidth=1)  # 调整线条粗细
plt.plot(trues_flat, label='True Values', linewidth=1)  # 调整线条粗细
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Predictions vs True Values')
plt.legend()
plt.show()
