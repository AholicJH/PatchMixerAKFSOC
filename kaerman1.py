import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class AKF:
    def __init__(self, dim):
        self.F = np.eye(dim)  # 状态转移矩阵
        self.H = np.eye(dim)  # 观测矩阵
        self.Q = np.eye(dim) * 0.001  # 初始过程噪声协方差
        self.R = np.eye(dim) * 1  # 初始观测噪声协方差
        self.P = np.eye(dim) * 1  # 估计误差协方差
        self.x = np.zeros((dim, 1))  # 初始状态

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)
        self.P = np.dot((np.eye(len(self.x)) - np.dot(K, self.H)), self.P)
        self.Q *= 1 + 0.01 * (np.dot(y.T, y) - self.Q)
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
train_data = data.iloc[:int(len(data) * 0.8)]
akf = AKF(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm = LSTMModel(3, 500, 4, 1).to(device)  # 将模型移动到GPU
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

predictions = []
akf_predictions = []

num_epochs = 20  # 增加训练轮数
for epoch in range(num_epochs):
    predictions = []
    akf_predictions = []
    total_loss = 0
    print_count = 0  # 初始化计数器

    for index, row in train_data.iterrows():
        inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)  # 将输入张量移动到GPU
        true_value = torch.tensor([[row['SOC']]]).to(device)  # 将真实值张量移动到GPU
        prediction = lstm(inputs)
        loss = criterion(prediction, true_value)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        akf.predict()
        # measurement = np.array([[prediction.item()]])
        measurement = np.array([[row['SOC']]])  # 使用真实的SOC值作为测量值
        adjusted_state = akf.update(measurement)

        predictions.append(prediction.item())
        akf_predictions.append(adjusted_state[0, 0])

        # 打印前五个真实值，预测值，AKF后的值
        if print_count < 5:
            print(
                f"True Value: {true_value.item()}, Prediction: {prediction.item()}, AKF Adjusted: {adjusted_state[0, 0]}")
            print_count += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_data)}")
plt.figure(figsize=(10, 5))
plt.plot(train_data['SOC'].values, label='True SOC')
plt.plot(predictions, label='LSTM Predictions')
plt.plot(akf_predictions, label='AKF Adjusted Predictions')
plt.legend()
plt.title('LSTM vs AKF Predictions vs True SOC')
plt.xlabel('Time Step')
plt.ylabel('SOC')
plt.show()
