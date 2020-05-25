# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:42:31 2020
FizzBuzz是一个简单的小游戏，游戏规则是：从1往上数树，当遇到3的倍数的时候，说fizz，当遇到5的倍数，说buzz，
当遇到15的倍数时，说fizzbuzz，其他情况下则正常数数。
写一个简单的小程序来确定要返回正常数值还是fizz、buzz或者fizzBuzz

@author: Administrator
"""
def fizz_buzz_encode(i):
    if i % 15 == 0 : return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0
def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]
def helper(i):
    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))
for i in range(1,115):
    helper(i)
    
    
