# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:17:38 2020
神经网络做FizzBuzz，让神经网络学会FizzBuzz
1、首先定义模型的输入与输出（训练数据）
2、然后用pytorch定义模型:
    - 为了让我们的模型学会FizzBuzz这个游戏，我们需要定义一个损失函数和一个优化算法；
    - 这个优化算法会不断优化损失函数，使得模型在该任务上取得尽可能低的损失值；
    - 损失值往往表示我们的模型表现好，损失值高表示我们的模型表现差；
    - 由于FizzBuzz游戏本质上是一个分类为你，我们选用Cross Entropyy Loss函数
    - 优化函数我们选用Stochastic Gradient Descent。
@author: Administrator
"""

import numpy as np
import torch
NUM_DIGITS = 10
def fizz_buzz_encode(i):
    if i % 15 == 0 : return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0
def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]
# 首先定义训练数据，一个二进制转换函数
# 因为这个模型直接用十进制数训练较难拟合，用二进制更好
def binary_encode(i,num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1]) # 后面[::-1]是为了倒序一下
# 把101~1024的数据作为训练数据
trX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(101,2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2 ** NUM_DIGITS)]) # Y是一个表示类别的数据，只能是0123

# 然后用PyTorch定义模型
NUM_HIDDEN = 100
model = torch.nn.Sequential(
        torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(NUM_HIDDEN,4) # 4 logits,after softmax,we get a probability distribution
        )
# 如果有GPU可用，就用GPU
if torch.cuda.is_available():
    model = model.cuda()
# 定义损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
# 模型训练
BATCH_SIZE = 128
for epoch in range(20000):
    for start in range (0,len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]
        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()
        y_pred = model(batchX)
        loss = loss_fn(y_pred,batchY)
        
        print("Epoch",epoch,loss.item())
        
        optimizer.zero_grad()
        loss.backward() # backpass
        optimizer.step() # gradient desent

# 用训练好的模型测试
testX = torch.Tensor([binary_encode(i,NUM_DIGITS) for i in range(1,101)])
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():
    testY = model(testX)
    
# predicts = zip(range(1,101),list(testY.max(1)(1)))
# 取最大值
predictions = zip(range(1, 101), testY.max(1)[1].data.tolist()) # zip打包成元组，tolist转成列表
# testY.max(1)取出第一个维度上的最大值，会返回两个tensor，
# 一个表示最大的数，另一个表示索引，我们这里要用索引，[1]取出索引
print([fizz_buzz_decode(i, x) for i, x in predictions])
    
