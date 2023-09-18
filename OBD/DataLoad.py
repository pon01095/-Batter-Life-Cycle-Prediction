# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:26:42 2021

@author: User
"""

import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import csv
# import Outlier
from matplotlib import font_manager, rc

"""
--------------------------------- 함수 ---------------------------------------
"""

def build_dataset(X, Y, seq_length):
    """ input과 output을 DataX와 DataY에 분류하는 함수"""
    
    DataX = []
    DataY = []
    
    X_Data = np.array(X)
    Y_Data = np.array(Y)
    
    for i in range(0, len(X_Data) - 2 * seq_length + 1):
        # Input (X)
        _x = X_Data[i:i+seq_length]
        DataX.append(_x)
        
        # Output (Y)
        _y = Y_Data[i + seq_length]
        DataY.append(_y)
    
    return np.array(DataX), np.array(DataY)


def Scaler(Dataset):
    
    MMScaler = MinMaxScaler()
    MinMax = MMScaler.fit(Dataset)
    
    return MinMax
    


Path = "C:\\Users\\tako\\Desktop\\EV log\\python code\\느시\\학습\\"
file_list = os.listdir(Path)


Test = pd.read_csv(Path + file_list[0], encoding='UTF-8')
Train = pd.read_csv(Path + file_list[1], encoding='UTF-8')

Test = Test.drop(['Unnamed: 0'], axis = 1)
Train = Train.drop(['Unnamed: 0'], axis = 1)

X_Train_bef, X_Test_bef, Y_Train, Y_Test = Train.drop(['SOC'], axis = 1), Test.drop(['SOC'], axis = 1), Train['SOC'], Test['SOC']


Scale = Scaler(X_Train_bef) # Min-Max Scaler
X_Train = Scale.transform(X_Train_bef) # Min-Max 정규화 (X 데이터)

Scale = Scaler(X_Test_bef) # Min-Max Scaler
X_Test = Scale.transform(X_Test_bef) # Min-Max 정규화 (X 데이터)

X_Data_Split = build_dataset(X_Train, Y_Train, 10) # Input, Output 데이터 분할
Y_Data_Split = build_dataset(X_Test, Y_Test, 10)
X_Train, X_Test, Y_Train, Y_Test = X_Data_Split[0], Y_Data_Split[0], X_Data_Split[1], Y_Data_Split[1]
