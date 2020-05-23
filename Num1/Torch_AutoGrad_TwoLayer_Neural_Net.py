# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:39:32 2020
改用torch实现一个两层神经网络,自动计算梯度
@author: Administrator
"""

import torch

N, D_in, H, D_out = 64, 1000, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H,requires_grad = True)
w2 = torch.randn(H, D_out,requires_grad = True)

learning_rate = 1e-6
for it in range(500):
    # 前向传输 Forward pass
    # h = x.mm(w1) # 得到 N * H
    # h_relu = h.clamp(min = 0) # N * H
    # y_pred = h_relu.mm(w2) # N * D_out
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)  # 简易写法
    # conpute loss 计算损失MSE均方误差
    loss = (y_pred - y).pow(2).sum() # computation graph计算图
    print(it,loss.item()) # 此时就不能再上一句中写item转为数值，否则的话就不能进行backward
    
    # Backward pass
    # computer the gradient,都是相对于loss求导
   
    loss.backward()
    
    
    # update weights of w1 and w2
    # 学习率的这个部分实际上也是一个计算图，为了不让这个计算图占内存，写 with torch.no_grad():
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_() # 要清零，否则会一直叠加
        w2.grad.zero_()

