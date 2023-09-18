#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:35:48 2021

@author: leehungi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

from mat2json import loadMat
from util import getBatteryCapacity, getChargingValues, getDischargingValues, getDataframe, series_to_supervised, rollingAverage, getSOH
from sklearn.metrics import mean_squared_error
from math import sqrt

# Battery_Data = pd.read_excel('nasa_data.xlsx')

# X = Battery_Data[['Capacity', 'Voltage_charge', 'Current_charge', 'Temperature_measured']]
# Y = Battery_Data[['Capacity']]

# B0005 = loadMat('B0005.mat')
# B0006 = loadMat('B0006.mat')
# B0007 = loadMat('B0007.mat')
# B0018 = loadMat('B0018.mat')

# dfB0005 = getDataframe(B0005)
# dfB0006 = getDataframe(B0006)
# dfB0007 = getDataframe(B0007)
# dfB0018 = getDataframe(B0018)

# X1 = dfB0005.iloc[:,[2,4]]
# X2 = dfB0006.iloc[:,[2,4]]
# X3 = dfB0007.iloc[:,[2,4]]
# Y1 = dfB0005.iloc[:,[6]]
# Y2 = dfB0006.iloc[:,[6]]
# Y3 = dfB0007.iloc[:,[6]]

# Xpred = dfB0018.iloc[:,[2,4]]
# Ypred = dfB0018.iloc[:,[6]]

# min_max_scaler = MinMaxScaler()
# Nor1 = min_max_scaler.fit(X1)
# X_Nor1 = Nor1.transform(X1)

# Nor2 = min_max_scaler.fit(X2)
# X_Nor2 = Nor2.transform(X2)

# Nor3 = min_max_scaler.fit(X3)
# X_Nor3 = Nor3.transform(X3)

# Nor4 = min_max_scaler.fit(Xpred)
# X_Nor4 = Nor3.transform(Xpred)

#Ambient temp 24
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')

# #Ambient temp 24 
# B0025 = loadMat('B0025.mat')
# B0026 = loadMat('B0026.mat')
# B0027 = loadMat('B0027.mat')
# B0028 = loadMat('B0028.mat')

# #Ambient temp 4 
# B0045 = loadMat('B0045.mat')
# B0046 = loadMat('B0046.mat')
# B0047 = loadMat('B0047.mat')
# B0048 = loadMat('B0048.mat')

# #Ambient temp 4
# B0053 = loadMat('B0053.mat')
# B0054 = loadMat('B0054.mat')
# B0055 = loadMat('B0055.mat')
# B0056 = loadMat('B0056.mat')

# B0005_SOH = getSOH(B0005_capacity)
# B0005_SOH = {'SOH' : B0005_SOH}
# B0005_SOH = pd.DataFrame(B0005_SOH)


dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
dfB0007 = getDataframe(B0007)
dfB0018 = getDataframe(B0018)

# dfB0025 = getDataframe(B0025)
# dfB0026 = getDataframe(B0026)
# dfB0027 = getDataframe(B0027)
# dfB0028 = getDataframe(B0028)

# dfB0045 = getDataframe(B0045)
# dfB0046 = getDataframe(B0046)
# dfB0047 = getDataframe(B0047)
# dfB0048 = getDataframe(B0048)

# # dfB0053 = getDataframe(B0053)
# # dfB0054 = getDataframe(B0054)
# dfB0055 = getDataframe(B0055)
# dfB0056 = getDataframe(B0056)

dfB0005 = dfB0005.fillna(0)
dfB0006 = dfB0006.fillna(0)
dfB0007 = dfB0007.fillna(0)
dfB0018 = dfB0018.fillna(0)

# dfB0025 = dfB0025.fillna(0)
# dfB0026 = dfB0026.fillna(0)
# dfB0027 = dfB0027.fillna(0)
# dfB0028 = dfB0028.fillna(0)

# dfB0045 = dfB0045.fillna(0)
# dfB0046 = dfB0046.fillna(0)
# dfB0047 = dfB0047.fillna(0)
# dfB0048 = dfB0048.fillna(0)

# dfB0053 = dfB0053.fillna(0)
# # dfB0054 = dfB0054.fillna(0)
# dfB0055 = dfB0055.fillna(0)
# dfB0056 = dfB0056.fillna(0)

X1 = dfB0005.iloc[:,[2,4]]
X2 = dfB0006.iloc[:,[2,4]]
X3 = dfB0007.iloc[:,[2,4]]
# X4 = dfB0025.iloc[:,[4]]
# X5 = dfB0026.iloc[:,[4]]
# X6 = dfB0027.iloc[:,[4]]
# X7 = dfB0028.iloc[:,[4]]
# X8 = dfB0045.iloc[:,[4]]
# X9 = dfB0046.iloc[:,[4]]
# X10 = dfB0047.iloc[:,[4]]
# X11 = dfB0048.iloc[:,[4]]
# X12 = dfB0055.iloc[:,[4]]
# X13 = dfB0056.iloc[:,[4]]

Y1 = dfB0005.iloc[:,[9]]
Y2 = dfB0006.iloc[:,[9]]
Y3 = dfB0007.iloc[:,[9]]
# Y4 = dfB0025.iloc[:,[7]]
# Y5 = dfB0026.iloc[:,[7]]
# Y6 = dfB0027.iloc[:,[7]]
# Y7 = dfB0028.iloc[:,[7]]
# Y8 = dfB0045.iloc[:,[7]]
# Y9 = dfB0046.iloc[:,[7]]
# Y10 = dfB0047.iloc[:,[7]]
# Y11 = dfB0048.iloc[:,[7]]
# Y12 = dfB0055.iloc[:,[7]]
# Y13 = dfB0056.iloc[:,[7]]

Xpred = dfB0018.iloc[:,[2,4]]
Ypred = dfB0018.iloc[:,[9]]

min_max_scaler = MinMaxScaler()
Nor1 = min_max_scaler.fit(X1)
X_Nor1 = Nor1.transform(X1)

Nor2 = min_max_scaler.fit(X2)
X_Nor2 = Nor2.transform(X2)

Nor3 = min_max_scaler.fit(X3)
X_Nor3 = Nor3.transform(X3)

# Nor4 = min_max_scaler.fit(X4)
# X_Nor4 = Nor4.transform(X4)

# Nor5 = min_max_scaler.fit(X5)
# X_Nor5 = Nor5.transform(X5)

# Nor6 = min_max_scaler.fit(X6)
# X_Nor6 = Nor6.transform(X6)

# Nor7 = min_max_scaler.fit(X7)
# X_Nor7 = Nor7.transform(X7)

# Nor8 = min_max_scaler.fit(X8)
# X_Nor8 = Nor8.transform(X8)

# Nor9 = min_max_scaler.fit(X9)
# X_Nor9 = Nor9.transform(X9)

# Nor10 = min_max_scaler.fit(X10)
# X_Nor10 = Nor10.transform(X10)

# Nor11 = min_max_scaler.fit(X11)
# X_Nor11 = Nor11.transform(X11)

# Nor12 = min_max_scaler.fit(X12)
# X_Nor12 = Nor12.transform(X12)

# Nor13 = min_max_scaler.fit(X13)
# X_Nor13 = Nor13.transform(X13)

Nor14 = min_max_scaler.fit(Xpred)
X_Nor14 = Nor14.transform(Xpred)


def build_dataset(X_Data, Y_Data, seq_length):
    """ input과 output을 DataX와 DataY에 분류하는 함수"""
    
    DataX = []
    DataY = []
    
    for i in range(0, len(X_Data) - seq_length):
        # Input (X)
        _x = X_Data[i:i+seq_length]
        DataX.append(_x)
        
        # Output (Y)
        _y = Y_Data.iloc[i+seq_length]
        DataY.append(_y)
        
    return np.array(DataX), np.array(DataY)

# def build_dataset2(X_Data, Y_Data, seq_length):
#     """ input과 output을 DataX와 DataY에 분류하는 함수"""
    
#     DataX = []
#     DataY = []
    
#     for i in range(0, len(X_Data) - seq_length):
#         # Input (X)
#         _x = X_Data[i:i+seq_length]
#         DataX.append(_x)
        
#         # Output (Y)
#         _y = Y_Data[i:i+seq_length]
#         DataY.append(_y)
        
#     return np.array(DataX), np.array(DataY)


def Build_GRU(GRUCell, Neurons, learning_rate):

    GRUModel = keras.Sequential()
    
    GRUModel.add(layers.GRU(GRUCell, return_sequences=True, input_shape=(n_input, n_features)))
    GRUModel.add(layers.GRU(GRUCell, return_sequences=True))
    GRUModel.add(layers.GRU(GRUCell, return_sequences=False))
    GRUModel.add(layers.Dense(Neurons, activation='relu'))
    GRUModel.add(layers.Dense(Neurons, activation='relu'))
    GRUModel.add(layers.Dense(Neurons, activation='relu'))
    GRUModel.add(layers.Dense(Neurons, activation='relu'))
    GRUModel.add(layers.Dense(Neurons, activation='relu'))
    GRUModel.add(layers.Dense(1))
    
    GRUModel.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return GRUModel

def Score(GRUCell, Neurons, learning_rate, epochs):
    
    Model = Build_GRU(int(GRUCell), int(Neurons), learning_rate)
    Model.fit(X_train, Y_train, epochs=int(epochs), verbose=0)
    
    Y_pred = Model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    
    return r2

n_features = 2 # 변수 개수
n_input = 5 # input 시퀀스 길이


Data_X1, Data_Y1 = build_dataset(X_Nor1, Y1, n_input)
Data_X2, Data_Y2 = build_dataset(X_Nor2, Y2, n_input)
Data_X3, Data_Y3 = build_dataset(X_Nor3, Y3, n_input)
# Data_X4, Data_Y4 = build_dataset(X_Nor4, Y4, n_input)
# Data_X5, Data_Y5 = build_dataset(X_Nor5, Y5, n_input)
# Data_X6, Data_Y6 = build_dataset(X_Nor6, Y6, n_input)
# Data_X7, Data_Y7 = build_dataset(X_Nor7, Y7, n_input)
# Data_X8, Data_Y8 = build_dataset(X_Nor8, Y8, n_input)
# Data_X9, Data_Y9 = build_dataset(X_Nor9, Y9, n_input)
# Data_X10, Data_Y10 = build_dataset(X_Nor10, Y10, n_input)
# Data_X11, Data_Y11 = build_dataset(X_Nor11, Y11, n_input)
# Data_X12, Data_Y12 = build_dataset(X_Nor12, Y12, n_input)
# Data_X13, Data_Y13 = build_dataset(X_Nor13, Y13, n_input)

Data_Xpred, Data_Ypred = build_dataset(X_Nor14, Ypred, n_input)

Data_X = np.concatenate((Data_X1, Data_X2, Data_X3), axis=0)
Data_Y = np.concatenate((Data_Y1, Data_Y2, Data_Y3), axis=0)

# Data_X, Data_Y = build_dataset2(X, Y, n_input)

print(np.shape(Data_X))
print(np.shape(Data_Y))

X_train, X_test, Y_train, Y_test = Data_X, Data_Xpred, Data_Y, Data_Ypred
    
# X_train, X_test, Y_train, Y_test = train_test_split(Data_X, Data_Y, test_size=0.2, shuffle=False)
print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

parameters = {'GRUCell': (10, 15), 'Neurons': (10, 20), 'learning_rate': (0.01, 0.1), 'epochs': (100, 200)}
BO = BayesianOptimization(f=Score, pbounds=parameters, verbose=2)
BO.maximize(init_points=5, n_iter=10)
print(BO.max)

Best_GRUCell = int(BO.max['params']['GRUCell'])
Best_Neurons = int(BO.max['params']['Neurons'])
Best_epochs = int(BO.max['params']['epochs'])
Best_learning_rate = BO.max['params']['learning_rate']

Model = Build_GRU(Best_GRUCell, Best_Neurons, Best_learning_rate)
Model.summary()

Training = Model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=Best_epochs)

History = pd.DataFrame(Training.history)
History.loc[:, ['loss', 'val_loss']].plot()
History.loc[:, ['mean_absolute_error', 'val_mean_absolute_error']].plot()

Y_pred = Model.predict(X_test)

plt.plot(list(range(len(Ypred)+1-len(Y_pred),len(Ypred)+1)), Y_pred, label="Prediction")
plt.plot(Ypred, label="Raw")
plt.xlabel("Cycle")
plt.ylabel("Capacity")
plt.legend()
plt.show()

R2 = r2_score(Y_test, Y_pred)
rms = sqrt(mean_squared_error(Y_test, Y_pred))