# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:52:05 2020
词向量
复现论文Distributed Representations of Words and Phrases and their Compositionalty中训练词向量的方法
实现Skip-gram模型，并且适用论文中noice contrastive sampling的目标函数
正例和反例
对一个中心词，每出现一个正确的周围词，都要出现K个错误的周围词，使得正确的周围词与中心词的距离越小越好，错误的周围词与中心词的距离越大越好。

1、准备数据，设定参数；
2、创建单词表，这里选取常见的MAX_VOCAN_SIZE个单词
3、实现Dataloader，一个Dataloader需要以下内容
    - 把所有text编码成数字，然后用subsampling预处理这些文字；
    - 保存vocabulary，单词count，normalized word frequency
    - 每个iteration sample一个中心词
    - 根据当前的中心词返回contex单词
    - 返回单词的counts
使用Pytorch dataloader的教程：
必须定义两个函数：__len__()和__get__()
4、定义pytorch模型
    - 实现__init__函数，定义输入输出
    - 实现forward函数，计算损失函数
5、开始训练
    - 优化器
@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()
# 为了保证可复现性，通常把seed都固定，这样每次训练的结果都相近
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

# 设定一些超参数hyper parameters
C = 2 # 只定义周围三个单词context word
K = 5 # 随机采样，number of negative samples 
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 3000
BATCH_SIZE = 32
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 15

def word_tokenize(text):# 分割成一个一个单词
    return text.split()

# 创建单词表
with open("text8/text8.train.txt","r") as fin:
    text = fin.read()
text = text.split()  # 分割成一个一个单词
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1)) # 拿出最常见的MAX_VOCAB_SIZE个单词，留出一个位置给unk
vocab["<unk>"] = len(text) - np.sum(list(vocab.values())) # 把剩余的部分作为不常见的，values表示的是一个单词出现了多少次
# 构建两个词汇的mapping
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
# 查看方法：list(word_to_idx.item())[:100]
# 获取每个单词的频率
word_counts = np.array([count for count in vocab.values()], dtype = np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.) # 论文中要求
word_freqs = word_freqs / np.sum(word_freqs) 
VOCAB_SIZE = len(idx_to_word)

# 实现Dataloader
class WordEmbeddingDataset(tud.Dataset): # 继承
    # 初始化
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text] # 不是很懂？
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
    
    def __len__(self):
        # 这个数据集一共有多少个item
        return len(self.text_encoded)
    def __getitem__(self, idx):
        center_word = self.text_encoded[idx] # 中心词
        pos_indices = list(range(idx - C)) + list(range(idx + 1, idx + C + 1)) # 周围次，即中心词的前C个和后C个
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 取余，为了避免周围词的索引超出范围
        pos_words = self.text_encoded[pos_indices] # 周围词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True) # 采样错误，负例采样
        return center_word, pos_words,neg_words
# 创建dataset和dataloader
dataset = WordEmbeddingDataset(text,word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset,batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)

# 尝试打印


# 定义pytorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vcab_size, embed_size):
        super(EmbeddingModel,self).__init__()
        
        self.vocab_size = vcab_size
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        
        
    def forward(self,input_labels, pos_labels, neg_labels):  # 中心词，正确的周围词，错误的周围词
        # input_labels : [batch_size]
        # pos_labels:[batch_size,(window_size * 2)]
        # neg_labels:[batch_size, (window_size * 2 * K)]
        
        input_embedding = self.in_embed(input_labels) # [batch_size , embed_size] 的tensor
        pos_embedding = self.in_embed(pos_labels) # [batch_size,(window_size * 2)， embed_size] 的tensor
        neg_embedding = self.in_embed(neg_labels) # [batch_size,(window_size * 2 * K), embed_size] 的tensor
        
        input_embedding = input_embedding.unsqueeze(2) # 为了方便使用bmm进行点乘，多添一个第2维度的信息，使其变为[batch_size, embed_size,1]的tensor
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window_size * 2), 1]
        pos_dot = pos_dot.squeeze(2) # 压缩第2维,现在pos_dot的shape变为[batch_size, (windows_size * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2) # [batch_size, (window_size * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1) # 这里注意不要自己写F.log(F.sigmoid)，可能出现数据爆炸的现象
        log_neg = F.logsigmoid(neg_dot).sum(1) # 论文公式，第一维度求和
        
        loss = log_pos + log_neg
        
        # 返回的是[batch_size]
        return -loss # 论文公式只是一个objective，最终是要梯度下降，梯度不断减小
    
    def input_embeddings(self): # 方便取出input_embedding
        return self.in_embed.weight.data.cpu().numpy()
    
    
# 定义模型以及把模型移动到GPU，如果有的话

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
    
# 开始训练
    
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
            
            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print("epoch", e, "iteration", i, loss.item())
        
       
# 模型评估
                
        


