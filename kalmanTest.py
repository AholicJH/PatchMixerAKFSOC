import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class AUKF:
    def __init__(self, dim):
        self.dim = dim
        self.F = np.eye(dim)  # 状态转移矩阵
        self.H = np.eye(dim)  # 观测矩阵
        self.Q = np.eye(dim) * 0.001  # 初始过程噪声协方差
        self.R = np.eye(dim) * 1  # 初始观测噪声协方差
        self.P = np.eye(dim) * 0.1  # 估计误差协方差
        self.x = np.zeros((dim, 1))  # 初始状态
        self.kappa = 3 - dim  # 超参数

    def sigma_points(self):
        lambda_ = self.kappa
        sigma_points = np.zeros((2 * self.dim + 1, self.dim))
        sigma_points[0] = self.x.flatten()
        sqrt_P = np.linalg.cholesky((self.dim + lambda_) * (self.P + np.eye(self.dim) * 1e-3))
        for i in range(self.dim):
            sigma_points[i + 1] = self.x.flatten() + sqrt_P[i]
            sigma_points[self.dim + i + 1] = self.x.flatten() - sqrt_P[i]
        return sigma_points

    def predict(self):
        sigma_points = self.sigma_points()
        x_pred = np.dot(self.F, sigma_points.T).T.mean(axis=0).reshape(-1, 1)
        P_pred = self.Q + np.cov(np.dot(self.F, sigma_points.T))
        self.x, self.P = x_pred, P_pred
        # print("P after predict:", self.P)  # 打印P的值
        return self.x

    def update(self, z):
        sigma_points = self.sigma_points()
        z_pred = np.dot(self.H, sigma_points.T).T.mean(axis=0).reshape(-1, 1)
        P_zz = self.R + np.cov(np.dot(self.H, sigma_points.T))
        P_xz = np.cov(sigma_points.T, np.dot(self.H, sigma_points.T))[:self.dim, self.dim:]
        K = np.dot(P_xz, np.linalg.inv(P_zz))
        y = z - z_pred
        self.x += np.dot(K, y)
        self.P -= np.dot(K, np.dot(P_zz, K.T))
        self.P += np.eye(self.dim) * 1e-3  # 确保P在每次更新后是正定的
        self.P = (self.P + self.P.T) / 2  # 确保P是对称的
        self.P = np.where(self.P < 1e-6, 1e-6, self.P)  # 确保P的所有元素都是正数
        self.Q *= 1 + 0.01 * (np.dot(y.T, y) - self.Q)
        # print("P after update:", self.P)  # 打印P的值
        return self.x


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


data = pd.read_csv('./dataset/A1-007-US06-0-20120813.csv')
scaler = MinMaxScaler()
features = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
target = ['SOC']
data[features + target] = scaler.fit_transform(data[features + target])


# 划分训练集和验证集
train_data = data.iloc[:int(len(data) * 0.7)]
val_data = data.iloc[int(len(data) * 0.7):int(len(data) * 0.9)]
test_data = data.iloc[int(len(data) * 0.9):]

aukf = AUKF(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = LSTMModel(3, 500, 4, 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

predictions = []
akf_predictions = []
losses = []
val_losses = []

num_epochs = 20
patience = 5  # 早停法的耐心值
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
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

    avg_loss = total_loss / len(train_data)
    losses.append(avg_loss)

    # 验证阶段
    lstm.eval()
    val_loss = 0
    with torch.no_grad():
        for index, row in val_data.iterrows():
            inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
            true_value = torch.tensor([[row['SOC']]]).to(device)
            prediction = lstm(inputs)
            loss = criterion(prediction, true_value)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_data)
    val_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, Val Loss: {avg_val_loss}")

    # 早停法逻辑
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(lstm.state_dict(), './lstm_model/lstm_model_best.pth')  # 保存最佳模型
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
lstm.load_state_dict(torch.load('./lstm_model/lstm_model_best.pth',weights_only=True))  # 加载最佳模型
lstm.eval()
test_predictions = []
test_akf_predictions = []
with torch.no_grad():
    for index, row in test_data.iterrows():
        inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        prediction = lstm(inputs)
        aukf.predict()
        measurement = np.array([[row['SOC']]])
        adjusted_state = aukf.update(measurement)

        test_predictions.append(prediction.item())
        test_akf_predictions.append(adjusted_state[0, 0])

# 绘制测试结果
plt.figure(figsize=(10, 5))
plt.plot(test_data['SOC'].values, label='True SOC')
plt.plot(test_predictions, label='LSTM Predictions')
plt.plot(test_akf_predictions, label='AUKF Adjusted Predictions')
plt.legend()
plt.title('LSTM vs AUKF Predictions vs True SOC (Test Set)')
plt.xlabel('Time Step')
plt.ylabel('SOC')
plt.show()