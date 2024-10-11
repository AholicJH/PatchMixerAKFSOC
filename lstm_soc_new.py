import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 如果输入是二维的，添加批量维度
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

data = pd.read_csv('./dataset/A1-007-US06-0-20120813.csv')
scaler = MinMaxScaler()
features = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
target = ['SOC']
data[features + target] = scaler.fit_transform(data[features + target])

# 划分训练集、验证集和测试集
train_data = data.iloc[:int(len(data) * 0.7)]
val_data = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9)]
test_data = data.iloc[int(len(data) * 0.9):]

# 转换为TensorDataset并使用DataLoader加载数据
train_dataset = TensorDataset(torch.tensor(train_data[features].values, dtype=torch.float32),
                              torch.tensor(train_data[target].values, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(val_data[features].values, dtype=torch.float32),
                            torch.tensor(val_data[target].values, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_data[features].values, dtype=torch.float32),
                             torch.tensor(test_data[target].values, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = LSTMModel(3, 100, 2, 1, dropout_prob=0.4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0005)

# 训练与验证过程
losses = []
val_losses = []

num_epochs = 20
patience = 3  # 早停法的耐心值
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # 训练模型
    lstm.train()
    total_loss = 0

    # 训练阶段
    for i, (inputs, true_value) in enumerate(train_loader):
        inputs, true_value = inputs.to(device), true_value.to(device).unsqueeze(1)
        true_value = true_value.squeeze(-1)  # 去掉多余的维度，使其与 prediction 匹配
        prediction = lstm(inputs)
        prediction = prediction.squeeze(-1)  # 去掉多余的维度，使其与 true_value 匹配
        loss = criterion(prediction, true_value)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if i < 5:  # 打印前5个真实值和预测值
            print(f"Validation - True Value: {true_value.detach().cpu().numpy()[:1]}, Prediction: {prediction.detach().cpu().numpy()[:5]}")

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    # 验证阶段
    lstm.eval()
    print("验证阶段")
    val_loss = 0
    # 验证阶段
    with torch.no_grad():
        for inputs, true_value in val_loader:
            inputs, true_value = inputs.to(device), true_value.to(device).unsqueeze(1)
            true_value = true_value.squeeze(-1)  # 保证尺寸一致
            prediction = lstm(inputs)
            loss = criterion(prediction, true_value)
            val_loss += loss.item()

            # # 打印整个批次的真实数据和预测数据
            # for i in range(len(true_value)):
            #     print(f"Validation - True Value: {true_value[i].item()}, Prediction: {prediction[i].item()}")

    avg_val_loss = val_loss / len(val_loader)
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
true_values = []

with torch.no_grad():
    for inputs, true_value in test_loader:
        inputs = inputs.to(device)
        prediction = lstm(inputs)
        test_predictions.extend(prediction.cpu().numpy())
        true_values.extend(true_value.cpu().numpy())

# 绘制测试结果
plt.figure(figsize=(10, 5))
plt.plot(test_data['SOC'].values, label='True SOC')
plt.plot(test_predictions, label='LSTM Predictions')
plt.legend()
plt.title('LSTM Predictions vs True SOC (Test Set)')
plt.xlabel('Time Step')
plt.ylabel('SOC')
plt.show()