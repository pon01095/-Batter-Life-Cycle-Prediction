# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:53:11 2022

@author: User
"""

import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import csv
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error 
from math import *


from DataLoad_different_car import *

class MachineLearning():

        def __init__(self, X_train, X_test, Y_train, Y_test):
            
            self.X_train = X_train
            self.X_test = X_test
            self.Y_train = Y_train
            self.Y_test = Y_test
            
        def Model_Builder(self, params):
            """ 모델 build """

            n_input = self.X_train[0].shape[0] # input sequence
            n_features = self.X_train[0].shape[1] # input features
            
            tf.random.set_seed(2)
            LSTM = keras.Sequential()
            LSTM.add(layers.LSTM(int(params['LSTMCell']), return_sequences=True, input_shape=(n_input, n_features), kernel_initializer='random_uniform'))
            
            numLSTM = int(params['Num of LSTM layers']) # GRU layer 수
            numDense = int(params['Num of Dense layers']) # Dens layer 수
            
            for i in range(numLSTM): # GRU layer
                
                LSTMCell = 'LSTMCell_{0}'.format(i+1)
                if i == numLSTM-1:
                    LSTM.add(layers.LSTM(int(params[LSTMCell]), return_sequences=False, kernel_initializer='random_uniform')) # kernel_initializer(가중치 초기화)='random_uniform(일정 구간 내에서 랜덤하게 찍는 방법)
                else:
                    LSTM.add(layers.LSTM(int(params[LSTMCell]), return_sequences=True, kernel_initializer='random_uniform'))
            
            LSTM.add(layers.Dense(int(params['Neurons']), activation='relu', kernel_initializer='random_uniform'))
            
            for j in range(numDense): # Dense layer
                
                Cell = 'Neurons_{0}'.format(j+1)
                LSTM.add(layers.Dense(int(params[Cell]), activation='relu', kernel_initializer='random_uniform'))
                
            LSTM.add(layers.Dense(1)) # Output layer
            LSTM.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']), loss='mean_squared_error', metrics=['mean_absolute_error'])
        
            return LSTM
    
        def Training(self, params, SavePath):
   
            start= time.time()
            MLModel = self.Model_Builder(params)
            MLModel.fit(self.X_train, self.Y_train, epochs=params['Epochs'], verbose=1, batch_size=64)
            end= time.time()
            
            Y_pred = MLModel.predict(self.X_test)
            print(Y_pred)
            R2 = r2_score(self.Y_test, Y_pred)
            print("R-squred = ", R2)
            print("최적화 시간: ", round(end-start, 3), "초")
            
            MLModel.save(SavePath)
        
            return Y_pred

"""
--------------------------- 아래 부분부터 수정 -------------------------------- 
"""
# Path = "C:\\Users\\tako\\Desktop\\EV log\\python code\\느시\\학습\\"
# file_list = os.listdir(Path)


# Test = pd.read_csv(Path + file_list[0], encoding='UTF-8')
# Train = pd.read_csv(Path + file_list[1], encoding='UTF-8')

# Test = Test.drop(['Unnamed: 0', 'Unnamed: 0.1', '날짜', '시간'], axis = 1)
# Train = Train.drop(['Unnamed: 0', 'Unnamed: 0.1', '날짜', '시간'], axis = 1)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

SavePath = "C:\\Users\\tako\\Desktop\\EV log\\python code\\best_Model_different_car"
params = {'Num of LSTM layers': 2, 'Num of Dense layers': 2, 'LSTMCell': 3, 'LSTMCell_1': 5, 'LSTMCell_2': 5, 
          'Neurons': 20, 'Neurons_1': 10, 'Neurons_2': 5, 'lr': 0.001, 'Epochs': 5}


# X_Train, X_Test, Y_Train, Y_Test = Train.drop(['SOC'], axis = 1), Test.drop(['SOC'], axis = 1), Train['SOC'], Test['SOC']

# X_Train = X_Train.reshape(X_Train.shape[0], X_Train[1], 1)

LSTM_Model = MachineLearning(X_Train, X_Test, Y_Train, Y_Test).Training(params, SavePath)

mean_squared_error(Y_Test, LSTM_Model)

plt.plot(LSTM_Model, label="Prediction", color = 'mediumblue')
plt.xlabel("Data point")
plt.ylabel("SOC")
plt.title('Total data of SOC')
plt.plot(Y_Test, color = 'deeppink', label = 'Raw')
plt.legend()
plt.show()
