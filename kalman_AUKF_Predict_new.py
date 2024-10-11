import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class AUKF:
    def __init__(self, dim,I, dt, Q_max):
        self.dim = dim
        self.I = I  # 电流
        self.dt = dt  # 时间步长
        self.Q_max = Q_max  # 最大电荷量
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

    def state_function(self, x):
        # 更新 SOC 值并加入过程噪声
        process_noise = np.random.multivariate_normal(np.zeros(self.dim), self.Q).reshape(-1, 1)
        # 更新 SOC 值
        return x - (self.I * self.dt / self.Q_max)+ + process_noise

    def predict(self):
        sigma_points = self.sigma_points()
        state_predictions = np.array([self.state_function(sp) for sp in sigma_points])
        state_predictions = state_predictions.reshape(state_predictions.shape[0], -1)  # Ensure 2D shape
        x_pred = state_predictions.mean(axis=0, keepdims=True).T
        P_pred = self.Q + np.cov(state_predictions.T)
        self.x, self.P = x_pred, P_pred
        return self.x

    def update(self, z):
        sigma_points = self.sigma_points()
        z_pred = np.array([np.dot(self.H, sp.reshape(-1, 1)) for sp in sigma_points]).mean(axis=0, keepdims=True).T
        z_points = np.array([np.dot(self.H, sp.reshape(-1, 1)) for sp in sigma_points]).reshape(-1, self.dim)
        P_zz = self.R + np.cov(z_points.T)
        # print(P_zz.shape)  # Should be (1, 1)

        # Ensure sigma_points and z_points have correct shapes
        sigma_points = sigma_points.reshape(self.dim, -1)
        z_points = z_points.reshape(self.dim, -1)

        # Compute P_xz correctly
        P_xz = np.cov(sigma_points, z_points, rowvar=True)[:self.dim, self.dim:]
        # print(P_xz.shape)  # Should be (1, 1)

        K = np.dot(P_xz, np.linalg.inv(P_zz))
        y = z - z_pred
        self.x += np.dot(K, y).reshape(self.x.shape)  # Adjust shape to match self.x
        self.P -= np.dot(K, np.dot(P_zz, K.T))
        self.P += np.eye(self.dim) * 1e-3  # Ensure P is positive definite after each update
        self.P = (self.P + self.P.T) / 2  # Ensure P is symmetric
        self.P = np.where(self.P < 1e-6, 1e-6, self.P)  # Ensure all elements of P are positive

        # Adjust the shape of np.dot(y.T, y) to match self.Q
        dot_product = np.dot(y.T, y).reshape(self.Q.shape)
        self.Q *= 1 + 0.01 * (dot_product - self.Q)

        return self.x

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def predict():
    # 加载归一化器
    scaler = joblib.load('./scaler/scaler.pkl')
    # 读取测试数据
    test_data = pd.read_csv('./dataset/A1-007-US06-0-20120813.csv')
    # scaler = MinMaxScaler()
    features = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1']
    target = ['SOC']
    test_data[features + target] = scaler.transform(test_data[features + target])
    # 初始化模型
    # I = 1.0  # Example current value, replace with actual value
    dt = 1.0  # Example time step, replace with actual value
    Q_max = 1.0  # Example maximum charge, replace with actual value
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm = LSTMModel(3, 10, 1, 1).to(device)

    # 加载模型权重
    try:
        lstm.load_state_dict(torch.load('./lstm_model/lstm_best.pth', weights_only=True))
        print("模型权重加载成功")
    except Exception as e:
        print(f"模型权重加载失败: {e}")
        return
    # lstm.load_state_dict(torch.load('./lstm_model/lstm_model.pth', weights_only=True))
    # print(lstm)
    lstm.eval()

    predictions = []
    akf_predictions = []
    noise_std =0.00001
    for index, row in test_data.iterrows():

        I = row['Current(A)']  # 获取当前步的电流值
        aukf = AUKF(1, I, dt, Q_max)  # 初始化AUKF模型

        inputs = torch.tensor(row[features].to_numpy(dtype=np.float32), dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)
        lstm_prediction = lstm(inputs)
        aukf.predict()
        measurement = row['SOC'] + np.random.normal(0, noise_std)  # SOC(k) + w
        adjusted_state = aukf.update(np.array([[measurement]]))

        predictions.append(lstm_prediction.item())
        akf_predictions.append(adjusted_state[0, 0])


        # print(f"True Value: {row['SOC']}, Prediction: {prediction.item()}, AUKF Adjusted: {adjusted_state[0, 0]}")


    plt.figure(figsize=(10, 5))
    plt.plot(test_data['SOC'].values, label='True SOC')
    plt.plot(predictions, label='LSTM Predictions')
    plt.plot(akf_predictions, label='AUKF Adjusted Predictions')
    plt.legend()
    plt.title('LSTM vs AUKF Predictions vs True SOC')
    plt.xlabel('Time Step')
    plt.ylabel('SOC')
    plt.show()

# 调用预测函数
predict()