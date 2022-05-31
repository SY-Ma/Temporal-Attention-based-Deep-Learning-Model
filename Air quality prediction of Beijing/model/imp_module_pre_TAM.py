# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : Air quality prediction of Beijing
# @File          : imp_module_pre_relu.py
# @Time          : 2022/4/3 17:15
# @Author        : SY.M
# @Software      : PyCharm


import math
import torch
from torch.nn import Module
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Imp_Single(Module):
    def __init__(self, num_feature,
                 time_len,
                 hidden_len,
                 q,
                 k,
                 v,
                 h,
                 N,
                 temp_len,
                 num_classes=None,
                 mask=True,
                 is_stand=True):
        super(Imp_Single, self).__init__()
        # self.standard_layer = torch.nn.InstanceNorm1d(num_features=num_feature)
        self.standard_layer = torch.nn.BatchNorm1d(num_features=num_feature)

        self.input_layer_feature = Input_Layer(num_feature=num_feature, in_features=time_len, hidden_len=hidden_len, is_stand=is_stand, pe=True)
        self.encoder_stack_feature = torch.nn.ModuleList([Encoder(hidden_len=hidden_len,
                                                          q=q, k=k, v=v, h=h,
                                                          mask=False, temp_len=temp_len,
                                                          in_channel=num_feature) for _ in range(N)])
        self.input_layer_time = Input_Layer(num_feature=num_feature, in_features=num_feature, hidden_len=hidden_len, is_stand=is_stand, pe=True)
        self.encoder_stack_time = torch.nn.ModuleList([Encoder(hidden_len=hidden_len,
                                                            q=q, k=k, v=v, h=h,
                                                            mask=False, temp_len=temp_len,
                                                          in_channel= time_len) for _ in range(N)])
        self.output_linear = Output_Layer(time_len=time_len, num_feature=num_feature, num_classes=num_classes, hidden_len=hidden_len)

    def forward(self, x, stage, feature_idx):
        global score_feature, score_time
        previous = x[:, feature_idx, -1].unsqueeze(-1)
        x = self.standard_layer(x)

        x_feature = self.input_layer_feature(x)
        x_time = self.input_layer_time(x.transpose(-1, -2))

        for encoder in self.encoder_stack_feature:
            x_feature, score_feature = encoder(x_feature, stage=stage)

        for encoder in self.encoder_stack_time:
            x_time, score_time = encoder(x_time, stage=stage)

        x = self.output_linear(x_time=(x_time, previous), x_feature=x_feature)

        return x, score_feature, score_time



class Input_Layer(Module):
    def __init__(self, num_feature, in_features, hidden_len, is_stand=True, pe=True):
        super(Input_Layer, self).__init__()
        self.pe = pe
        self.input_linear = torch.nn.Linear(in_features=in_features, out_features=hidden_len)

    def forward(self, x):

        x = self.input_linear(x)

        if self.pe:
            x = self.position_encode(x)

        return x

    def position_encode(self, x):
        pe = torch.ones_like(x[0])
        position = torch.arange(0, x.shape[1]).unsqueeze(-1)
        temp = torch.Tensor(range(0, x.shape[-1], 2))
        temp = temp * -(math.log(10000) / x.shape[-1])
        temp = torch.exp(temp).unsqueeze(0)
        temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        pe[:, 0::2] = torch.sin(temp)
        pe[:, 1::2] = torch.cos(temp)

        return x + pe

class Encoder(Module):
    def __init__(self, in_channel, hidden_len, q, k, v, h, mask, temp_len=512):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(hidden_len=hidden_len, q=q, k=k, v=v, h=h, mask=mask)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, stage):

        x, score = self.MHA(x, stage=stage)
        x = self.relu(x)
        x = self.dropout(x)

        return x, score


class MultiHeadAttention(Module):
    def __init__(self, hidden_len, q, k, v, h, mask=True):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = torch.nn.Linear(in_features=hidden_len, out_features=h * q)
        self.W_K = torch.nn.Linear(in_features=hidden_len, out_features=h * k)
        self.W_V = torch.nn.Linear(in_features=hidden_len, out_features=h * v)

        self.mask = mask
        self.h = h
        self.inf = -2**32+1

        self.out_linear = torch.nn.Linear(v * h, hidden_len)

    def forward(self, x, stage):

        Q = torch.cat(torch.chunk(self.W_Q(x), self.h, dim=-1), dim=0)
        K = torch.cat(torch.chunk(self.W_K(x), self.h, dim=-1), dim=0)
        V = torch.cat(torch.chunk(self.W_V(x), self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(DEVICE))

        score = torch.softmax(score, dim=-1)

        attention = torch.cat(torch.chunk(torch.matmul(score, V), self.h, dim=0), dim=-1)

        out = self.out_linear(attention)

        return out, score


class Output_Layer(Module):
    def __init__(self,
                 time_len,
                 num_feature,
                 hidden_len,
                 num_classes=None):
        super(Output_Layer, self).__init__()

        # 双塔
        self.time_linear = torch.nn.Linear(in_features=time_len * hidden_len, out_features=1)
        self.feature_linear = torch.nn.Linear(in_features=num_feature * hidden_len, out_features=1)
        self.weight_linear = torch.nn.Parameter(torch.Tensor([0.33, 0.33, 0.33, 0]))



    def forward(self, x_time, x_feature):

        previous = x_time[1]
        x_time = x_time[0]

        x_time = self.time_linear(x_time.reshape(x_time.shape[0], -1))
        x_feature = self.feature_linear(x_feature.reshape(x_feature.shape[0], -1))
        x = x_time * self.weight_linear[0] + x_feature * self.weight_linear[1] + self.weight_linear[3] + previous * self.weight_linear[2]

        return x


