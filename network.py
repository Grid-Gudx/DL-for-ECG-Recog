# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:42:16 2021

@author: gdx
"""

from tensorflow.keras.layers import *
from tensorflow.keras import Model

def cnn_block(x, filters, kernel_size): 
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Activation('relu')(x)   
    # x = BatchNormalization(axis=-1,trainable=False)(x) #对每个通道进行归一化
    x = MaxPooling1D(2)(x)
    return x

def res_block(x, filters, kernel_size):
    skip = Conv1D(filters, 1, padding='same')(x)
    
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    
    x = add([skip, x])
    x = Activation('relu')(x)   
    # x = BatchNormalization(axis=-1,trainable=False)(x) #对每个通道进行归一化
    x = MaxPooling1D(2)(x)
    return x


def cnn(x, output_dim=8):
    x = cnn_block(x, 8, 3)
    x = cnn_block(x, 16, 3)
    x = cnn_block(x, 32, 3)
    x = cnn_block(x, 64, 3)
    
    x = Flatten()(x)
    x = Dropout(0.2)(x)   
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='softmax')(x)
    return x

def resnet(x, output_dim=8):
    x = res_block(x, 8, 3)
    x = res_block(x, 16, 3)
    x = res_block(x, 32, 3)
    x = res_block(x, 64, 3)
    
    x = Flatten()(x)
    x = Dropout(0.2)(x)   
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='softmax')(x)
    return x

def model_creat(input_dim=48, output_dim=8, model='cnn'):
    input_shape = (input_dim,1) # input time_series dimensions
    input_tensor = Input(shape=input_shape)
    if model == 'cnn':
        output_tensor = cnn(input_tensor, output_dim)
    elif model == 'resnet':
        output_tensor = resnet(input_tensor, output_dim)
    else:
        print('model type is error')
    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model

if __name__ == '__main__':
    model = model_creat(input_dim=205, output_dim=4, model='resnet')
    model.summary()
    model.save('./model_struct/resnet.h5')
