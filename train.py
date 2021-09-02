# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:01:03 2021

@author: gdx
"""

import numpy as np
from sklearn.preprocessing import label_binarize
from tensorflow.keras import optimizers
from utils import mc_metrics, LossHistory
from network import model_creat

### load data
train_path = './data/dataset/train_data_1.npy'
val_path = './data/dataset/val_data_1.npy'
time_points = 205
num_classes = 4 #type num
train_data = np.load(train_path)
val_data = np.load(val_path)

x_train = train_data[:,0:-1].reshape(-1,time_points,1)
y_train = label_binarize(train_data[:,-1], list(range(num_classes)))

x_val = val_data[:,0:-1].reshape(-1,time_points,1)
y_val = label_binarize(val_data[:,-1], list(range(num_classes)))

### train model
batch_size = 256 #批尺寸，即一次训练所选取的样本数，batch_size=1时为在线学习
epochs = 50 #训练轮数
model_type = 'resnet' # cnn, lstm
model_save_path='./model/' + model_type + '2.h5'

model = model_creat(input_dim=time_points, output_dim=num_classes, model=model_type)
model.summary()

myhistory = LossHistory(model_path=model_save_path)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),#'Adam',
              metrics=['categorical_accuracy'])

#model_fit
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=0, #进度条
          validation_data=(x_val, y_val),#验证集
          callbacks=[myhistory])

#plot acc-loss curve
myhistory.loss_plot()