# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:39:32 2020
改用torch实现一个两层神经网络,手动计算梯度
@author: Administrator
"""

import torch

N, D_in, H, D_out = 64, 1000, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learning_rate = 1e-6
for it in range(500):
    # 前向传输 Forward pass
    h = x.mm(w1) # 得到 N * H
    h_relu = h.clamp(min = 0) # N * H
    y_pred = h_relu.mm(w2) # N * D_out
    
    # conpute loss 计算损失MSE均方误差
    loss = (y_pred - y).pow(2).sum().item()
    print(it,loss)
    
    # Backward pass
    # computer the gradient,都是相对于loss求导
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    
    
    # update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

