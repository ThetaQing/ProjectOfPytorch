# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:39:32 2020
改用torch实现一个两层神经网络,用NN模块实现，自动优化,封装成Class
小结：
1、定义输入输出
2、定义一个模型
3、定义损失函数
4、将model交给optimizer来做优化
5、进入训练过程：计算预测值；计算预测值与实际值的loss；做backward(),即后向传播；最后optimize参数.
@author: Administrator
"""

import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
class TwoLayerNet(torch.nn.Module):  # 继承
    def __init__(self,D_in,H,D_out):
        # 定义模型框架
        super(TwoLayerNet,self).__init__()  # 用super的方法初始化
        # 把每一个有导数的层都写在这里，定义模型框架
        self.linear1 = torch.nn.Linear(D_in, H,bias = False)  # w_1 * x + b_1
        self.linear2 = torch.nn.Linear(H,D_out,bias = False)
    def forward(self,x):  # 之前直接调用model(x)实际上就是调用model.forward(x)，相当于一个别称
        # 定义前向传播
        y_pred = self.linear2(self.linear1(x).clamp(min = 0))
        return y_pred
model = TwoLayerNet(D_in,H,D_out)
   
loss_fn = nn.MSELoss()
# Adam模型优化
learning_rate = 1e-4 # 对于Adam，1e-3~1e-4是比较好的学习率
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

for it in range(500):
    # Forward pass
    y_pred = model(x) # model.forward()
    # computer loss
    loss = loss_fn(y_pred,y)
    print(it,loss.item()) # 此时就不能再上一句中写item转为数值，否则的话就不能进行backward
    # 求导之前做一次清空
    optimizer.zero_grad()
    # Backward pass

    loss.backward()
    
    # update model parameters
    optimizer.step()