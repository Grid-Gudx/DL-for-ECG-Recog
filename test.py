# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:24:31 2021

@author: gdx
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from utils import mc_metrics

# df_testA = pd.read_csv('./data/testA.csv')

# id_ = np.array(df_testA['id'])

# ### load data
# test_path = './data/test_data.npy'
# time_points = 205

# test_data = np.load(test_path)

# x_test = test_data.reshape(-1,time_points,1)

# model = load_model('./model/resnet1.h5',compile=False)
# y_score = model.predict(x_test)

# label = np.zeros((20000, 4), dtype=int)
# idx = y_score.argmax(axis=1)

# for i in range(20000):
#     label[i, idx[i]]=1

# result = np.concatenate((id_.reshape(-1,1), label),axis=1)
# df_result = pd.DataFrame(result, columns=['id','label_0','label_1','label_2','label_3'])
# df_result.to_csv('./data/result2.csv', index=False)


val_path = './data/dataset/val_data_1.npy'
time_points = 205
num_classes = 4 #type num
val_data = np.load(val_path)

x_val = val_data[:,0:-1].reshape(-1,time_points,1)
y_val = label_binarize(val_data[:,-1], list(range(num_classes)))

model = load_model('./model/resnet1.h5',compile=False)
y_score = model.predict(x_val)

metric = mc_metrics(input_label_type='one_hot')
acc = metric.acc(y_score, y_val)
print(acc)
metric.confusion_plot(y_score, y_val)