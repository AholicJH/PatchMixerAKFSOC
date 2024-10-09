__all__ = ['PatchMixer']

# Cell
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.PatchTST_layers import *
from layers.RevIN import RevIN



'''
模型使用了一种称为 PatchMixer 的方法来处理时间序列数据
'''


'''
一个神经网络层，包含一个 ResNet 和一个 1x1 卷积层。
它的作用是对输入数据进行特征提取和变换
'''
class PatchMixerLayer(nn.Module):
    def __init__(self,dim,a,kernel_size = 8):
        super().__init__()
        self.Resnet =  nn.Sequential(
            nn.Conv1d(dim,dim,kernel_size=kernel_size,groups=dim,padding='same'),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim,a,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(a)
        )
    def forward(self,x):
        x = x +self.Resnet(x)                  # x: [batch * n_val, patch_num, d_model]
        x = self.Conv_1x1(x)                   # x: [batch * n_val, a, d_model]
        return x

'''
是整个模型的封装，包含一个Backbone模型
'''
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = Backbone(configs)
    def forward(self, x):
        x = self.model(x)
        return x

'''
使用自适应卡尔曼滤波器对模型的输出进行处理
'''
class AKF(nn.Module):
    def __init__(self, state_dim, measure_dim):
        super(AKF, self).__init__()
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.F = nn.Parameter(torch.eye(state_dim))  # 状态转移矩阵
        self.H = nn.Parameter(torch.eye(measure_dim, state_dim))  # 观测矩阵
        self.Q = nn.Parameter(torch.eye(state_dim))  # 过程噪声协方差
        self.R = nn.Parameter(torch.eye(measure_dim))  # 观测噪声协方差
        self.P = nn.Parameter(torch.eye(state_dim))  # 估计误差协方差
        self.x = nn.Parameter(torch.zeros(state_dim, 1))  # 初始状态

    def forward(self, z):
        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.t() + self.Q

        # 更新
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.t() + self.R
        K = P_pred @ self.H.t() @ torch.inverse(S)
        self.x = x_pred + K @ y
        self.P = (torch.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x
'''
Backbone模型，包含了多个PatchMixer层和一个LSTM层
'''
class Backbone(nn.Module):
    def __init__(self, configs,revin = True, affine = True, subtract_last = False):
        super().__init__()

        self.nvals = configs.enc_in       #输入的特征的数量
        self.lookback = configs.seq_len   #输入序列的长度
        self.forecasting = configs.pred_len  # 预测序列的长度
        self.patch_size = configs.patch_len  # 每个patch的长度
        self.stride = configs.stride  # patch的步长
        self.kernel_size = configs.mixer_kernel_size #PatchMixer层的卷积核大小

        self.PatchMixer_blocks = nn.ModuleList([])  #存储多个PatchMixer层
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))  # 用于填充patch
        self.patch_num = int((self.lookback - self.patch_size)/self.stride + 1) + 1 # 计算patch的数量
        # if configs.a < 1 or configs.a > self.patch_num:
        #     configs.a = self.patch_num
        self.a = self.patch_num   # PatchMixerLayer的参数a
        self.d_model = configs.d_model   # 模型的维度
        self.dropout = configs.dropout   # dropout率
        self.head_dropout = configs.head_dropout # head层的dropout率
        self.depth = configs.e_layers    # PatchMixerLayer的层数
        for _ in range(self.depth):
            self.PatchMixer_blocks.append(PatchMixerLayer(dim=self.patch_num, a=self.a, kernel_size=self.kernel_size))
        self.W_P = nn.Linear(self.patch_size, self.d_model)
        # 加入LSTM层
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=6, batch_first=True,
                            dropout=self.dropout)
        # self.akf = AKF(state_dim=self.d_model, measure_dim=self.d_model)  # 初始化AKF层
        # self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True,
        #                     dropout=self.dropout, bidirectional=True)
        self.head0 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * self.d_model, self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.head1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.a * self.d_model, int(self.forecasting * 2)),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(int(self.forecasting * 2), self.forecasting),
            nn.Dropout(self.head_dropout)
        )
        self.dropout = nn.Dropout(self.dropout)
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.nvals, affine=affine, subtract_last=subtract_last)
    def forward(self, x):
        bs = x.shape[0]
        nvars = x.shape[-1]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)                                                       # x: [batch, n_val, seq_len]

        x_lookback = self.padding_patch_layer(x)
        x = x_lookback.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # x: [batch, n_val, patch_num, patch_size]  

        x = self.W_P(x)                                                              # x: [batch, n_val, patch_num, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))      # x: [batch * n_val, patch_num, d_model]
        x = self.dropout(x)
        u = self.head0(x)

        for PatchMixer_block in self.PatchMixer_blocks:
            x = PatchMixer_block(x)
        # 加入LSTM网略
        x, _ = self.lstm(x)  # LSTM 前向传播
        x = self.head1(x)
        x = u + x
        x = torch.reshape(x, (bs , nvars, -1))                                       # x: [batch, n_val, pred_len]
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        # 在结果上应用 AKF
        # x = self.akf(z)  # AKF 前向传播
        return x