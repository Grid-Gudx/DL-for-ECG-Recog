# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:33:12 2021

@author: gdx
"""

import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt

# 加载训练集和测试集
df_train = pd.read_csv('./data/train.csv')
df_testA = pd.read_csv('./data/testA.csv')

### 转化为numpy
f = lambda x: list(map(float, x.split(','))) #input str out list

data = np.array(list(df_train['heartbeat_signals'].apply(f)))
label = np.array(df_train['label'])

data_test = np.array(list(df_testA['heartbeat_signals'].apply(f)))

#检查缺失值、数据格式等
np.isnan(data).sum()
data = data.astype(np.float32)
label = label.astype(np.float32)

np.isnan(data_test).sum()
data_test = data_test.astype(np.float32)

train_data = np.concatenate((data,label.reshape(-1,1)),axis=1)
np.save('./data/train_data.npy', train_data)
test_data = data_test
np.save('./data/test_data.npy', test_data)

### 确认标签列的各类别数量
def class_ratio(label, distribution=True):
    """
    Parameters
    ----------
    label : np.array([1,2,3,..]) 1D array
        DESCRIPTION.
    distribution : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    cl = np.unique(label)
    n = label.size #sample num
    for i in cl:
        m = np.sum(label==i)
        print(i, m, m/n)
    print('num ',n)
    if distribution:
        sns.distplot(label)

class_ratio(label)


### 分离验证集
from sklearn.model_selection import StratifiedKFold
def val_split(data_path, out_path):
    data = np.load(data_path)
    x = data[:, 0:-1]
    y = data[:, -1]
    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False) #分层交叉验证
    i = 0
    for train_index, test_index in skf.split(x, y):
        i += 1
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_data = np.concatenate((x_train, y_train.reshape(-1,1)),axis=1)
        val_data = np.concatenate((x_test, y_test.reshape(-1,1)),axis=1)
        np.save(out_path+'/train_data_'+str(i)+'.npy',train_data)
        np.save(out_path+'/val_data_'+str(i)+'.npy',val_data)
        
data_path = './data/train_data.npy'
out_path = './data/dataset'
val_split(data_path, out_path)
  
