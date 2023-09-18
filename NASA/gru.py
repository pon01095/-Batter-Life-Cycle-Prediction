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
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from keras.wrappers.scikit_learn import KerasRegressor

Battery_Data = pd.read_excel('nasa_data.xlsx')

X = Battery_Data[['Capacity', 'Voltage_charge', 'Current_charge', 'Temperature_measured']]
Y = Battery_Data[['Capacity']]

min_max_scaler = MinMaxScaler()
Nor = min_max_scaler.fit(X)
X_Nor = Nor.transform(X)

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

n_features = 4 # 변수 개수
n_input = 10 # input 시퀀스 길이

Data_X, Data_Y = build_dataset(X_Nor, Y, n_input)

print(np.shape(Data_X))
print(np.shape(Data_Y))


    
X_train, X_test, Y_train, Y_test = train_test_split(Data_X, Data_Y, test_size=0.2, shuffle=False)
print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

parameters = {'GRUCell': (10, 15), 'Neurons': (10, 20), 'learning_rate': (0.005, 0.1), 'epochs': (100, 200)}
BO = BayesianOptimization(f=Score, pbounds=parameters, verbose=2)
BO.maximize(init_points=5, n_iter=2)
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

plt.plot(list(range(len(Y)+1-len(Y_pred),len(Y)+1)), Y_pred, label="Prediction")
plt.plot(Y, label="Raw")
plt.xlabel("Cycle")
plt.ylabel("Capacity")
plt.legend()
plt.show()