# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:33:12 2021

@author: gdx
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载训练集和测试集
df_train = pd.read_csv('./data/train.csv')
df_testA = pd.read_csv('./data/testA.csv')

# 确认标签列的各类别数量
x = df_train['label']
print(df_train['label'].value_counts())
sns.distplot(x);

#转化为numpy
f = lambda x: list(map(float, x.split(','))) #input str out list

data = np.array(list(df_train['heartbeat_signals'].apply(f)))
label = np.array(df_train['label'])

#检查缺失值、数据格式等
np.isnan(data).sum()


data = data.astype(np.float32)
label = label.astype(np.float32)


def acc(_input, target):
    target=target.squeeze()
    acc=float((_input.argmax(dim=1) == target).float().mean())
    return acc



    




