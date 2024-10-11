import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,dropout=dropout_prob, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

data = pd.read_csv('./dataset/A1-007-US06-0-20120813.csv')
scaler = MinMaxScaler()
features = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
target = ['SOC']
data[features + target] = scaler.fit_transform(data[features + target])

# 保存归一化器
# joblib.dump(scaler, './scaler/scaler.pkl')

# 划分训练集和验证集
# train_data = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9)]
train_data = data.iloc[:int(len(data) * 0.7)]
val_data = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9)]
# val_data = data.iloc[:int(len(data) * 0.7)]
test_data = data.iloc[int(len(data) * 0.9):]
# test_data = data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = LSTMModel(3, 10, 1, 1, dropout_prob=0).to(device)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()  # 切换到MAE，对所有误差平等对待
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0001, weight_decay=1e-4)

losses = []
val_losses = []

num_epochs = 200
patience = 3 # 早停法的耐心值
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    #训练模型
    # 测试阶段
    print("训练阶段")
    lstm.train()
    total_loss = 0

    for index, row in train_data.iterrows():
        inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        true_value = torch.tensor([[row['SOC']]]).to(device)
        prediction = lstm(inputs)
        loss = criterion(prediction, true_value)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # 打印索引为500的倍数的真实数据和预测数据
        if index % 500 == 0:
            print( f"Epoch [{epoch + 1}/{num_epochs}], Index: {index}, True Value: {true_value.item()}, Prediction: {prediction.item()}")
    avg_loss = total_loss / len(train_data)
    losses.append(avg_loss)

    # 验证阶段
    lstm.eval()
    print("验证阶段")
    val_loss = 0
    with torch.no_grad():
        for index, row in val_data.iterrows():
            inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
            true_value = torch.tensor([[row['SOC']]]).to(device)
            prediction = lstm(inputs)
            loss = criterion(prediction, true_value)
            val_loss += loss.item()
            # 打印索引为500的倍数的真实数据和预测数据

            if index % 200 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Index: {index}, True Value: {true_value.item()}, Prediction: {prediction.item()}")

    avg_val_loss = val_loss / len(val_data)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, Val Loss: {avg_val_loss}")

    # 早停法逻辑
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(lstm.state_dict(), './lstm_model/lstm_best.pth')  # 保存最佳模型
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("早停法触发，停止训练")
            break

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 测试阶段
print("测试阶段")
lstm.load_state_dict(torch.load('./lstm_model/lstm_best.pth',weights_only=True))  # 加载最佳模型
lstm.eval()
test_predictions = []
with torch.no_grad():
    for index, row in test_data.iterrows():
        inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        prediction = lstm(inputs)
        test_predictions.append(prediction.item())
        if index % 200 == 0:
           print(f"Index: {index}, True Value: {row['SOC']}, Prediction: {prediction.item()}")


# 绘制测试结果
plt.figure(figsize=(10, 5))
plt.plot(test_data['SOC'].values, label='True SOC')
plt.plot(test_predictions, label='LSTM Predictions')
plt.legend()
plt.title('LSTM Predictions vs True SOC (Test Set)')
plt.xlabel('Time Step')
plt.ylabel('SOC')
plt.show()