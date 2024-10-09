import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def kalman_filter_predict(x_est, P, F, Q):
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kalman_filter_update(x_pred, P_pred, z, H, R, Q, adaptation_factor):
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred
    # Adaptive adjustment for the process noise covariance Q based on the magnitude of the innovation
    Q *= 1 + adaptation_factor * (np.dot(y.T, y) - Q)
    return x_upd, P_upd, Q


np.random.seed(0)
n_steps = 60
true_values = np.sin(np.linspace(0, 3 * np.pi, n_steps))
observations = true_values + np.random.normal(0, 0.5, n_steps)

model = LSTMModel(1, 10, 1, 1)

# Initialize Kalman Filter parameters
F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[0.001]])
R = np.array([[0.25]])
x_est = np.array([[0]])
P = np.array([[1]])
adaptation_factor = 0.01  # Control how fast the adaptation occurs

lstm_predictions = []
kf_adjusted_predictions = []

for i in range(n_steps - 1):
    current_input = torch.tensor([[observations[i]]], dtype=torch.float).reshape(-1, 1, 1)
    lstm_prediction = model(current_input).data.numpy().flatten()[0]
    lstm_predictions.append(lstm_prediction)

    x_pred, P_pred = kalman_filter_predict(x_est, P, F, Q)
    z = np.array([[observations[i + 1]]])
    x_upd, P_upd, Q = kalman_filter_update(x_pred, P_pred, z, H, R, Q, adaptation_factor)
    kf_adjusted_predictions.append(x_upd.flatten()[0])

    x_est, P = x_upd, P_upd

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(true_values[1:], label='True Values', linestyle='--')
plt.plot(lstm_predictions, label='LSTM Predictions')
plt.plot(kf_adjusted_predictions, label='AKF Adjusted Predictions')
plt.legend()
plt.title('Comparison of LSTM Predictions, AKF Adjusted Predictions, and True Values')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.grid(True)
plt.show()
