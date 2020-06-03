# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:09:07 2020

@author: Administrator
"""

import torch
x = torch.tensor(1,requires_grad = True) # 梯度默认False
w = torch.tensor(2,requires_grad = True)
b = torch.tensor(3,requires_grad = True)
y = w * x + b # 三个变量必须同一类型
y.backward()
print(w.grad)
print(x.grad)
print(b.grad)