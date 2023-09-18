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
    


Path =  "C:\\Users\\tako\\Desktop\\EV log\\카탈로그\\defined_data\\"

file_list = os.listdir(Path)

Train = pd.DataFrame()
for file_name in file_list:
    if '쏘울' in file_name:
        # print(i)
        print('test ' + file_name)
        Test = pd.read_csv(Path + file_name, encoding='UTF-8')
        

    else:
        print('Train ' + file_name)
        new = pd.read_csv(Path + file_name, encoding='UTF-8')
        
        Train = pd.concat([Train,new])
        
    # if i < len(file_list)-1:
    #     print(i)
    #     new = pd.read_csv(Path + file_list[i], encoding='UTF-8')
        
    #     Train = pd.concat([Train,new])
    # else:
    #     print('hi' + str(i))
    #     Test = pd.read_csv(Path + file_list[i], encoding='UTF-8')

# Test = Test.drop([0, 1])
# Test.reset_index(drop=True, inplace=True)

Train = Train.drop(['Unnamed: 0','Unnamed: 0.1', '날짜', '시간'], axis = 1)
Train = Train.fillna(0)
Test = Test.drop(['Unnamed: 0','Unnamed: 0.1', '날짜', '시간'], axis = 1)
Test = Test.fillna(0)

#쏘울의 경우
# Test.drop([0, 1])
# Test.reset_index(drop=True, inplace=True)

X_Data_Train = Train.drop(['SOC'], axis = 1)
Y_Data_Train = Train['SOC']

X_Data_Test = Test.drop(['SOC'], axis = 1)
Y_Data_Test = Test['SOC']


Scale = Scaler(X_Data_Train) # Min-Max Scaler
X_Train_MinMax = Scale.transform(X_Data_Train) # Min-Max 정규화 (X 데이터)

Scale = Scaler(X_Data_Test) # Min-Max Scaler
X_Test_MinMax = Scale.transform(X_Data_Test) # Min-Max 정규화 (X 데이터)


X_Train, Y_Train = build_dataset(X_Train_MinMax, Y_Data_Train, 10)

X_Test, Y_Test = build_dataset(X_Test_MinMax, Y_Data_Test, 10)

