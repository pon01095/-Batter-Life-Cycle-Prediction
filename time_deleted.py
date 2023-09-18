# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:37:16 2022

@author: tako
"""


import pandas as pd 
import matplotlib.pyplot as plt 
import os
import numpy as np
import csv

'''10초 단위로 data 뽑기'''

save_path = "C:\\Users\\tako\\Desktop\\EV log\\column_data\\time_deleted\\"
path = "C:\\Users\\tako\\Desktop\\EV log\\column_data\\"
file_list = os.listdir(path)

for file_name in file_list:
    if 'time_deleted' in file_name:  
        pass
    else:
        total_data = pd.read_csv(path + file_name, encoding='UTF-8')
        data =  total_data.drop([0])
        # print(data)
        # data = data.filter(regex='(?i)^(?!NaN).+', axis=1)
        # a = []
        
        # for i in range(len(total_data['SOC'])-1):
        #     if total_data['날짜'][i] == total_data['날짜'][i+1]:
        #         if total_data['시간'][i] == total_data['시간'][i+1]:
        #            a.append(i)
        a = []
    
        for i in range(len(total_data['SOC'])-1):
            if total_data['날짜'][i] == total_data['날짜'][i+1]:
                if len(total_data['시간'][i]) == 8:
                    if total_data['시간'][i][0:7] == total_data['시간'][i+1][0:7]:
                        a.append(i)
                if len(total_data['시간'][i]) == 7:
                    if total_data['시간'][i][0:6] == total_data['시간'][i+1][0:6]:
                        a.append(i)
                                    
        total_data2 = total_data.drop(a, axis = 0)       
        # total_data2.reset_index(drop=True, inplace=True)
        
        
        # total_data2 = total_data2.drop(['Unnamed: 0', '날짜', '시간', 'Temp 2', 'Temp 3', 'Temp 4',
        # 'Temp 5', 'Max Cell V No    ','Min Cell V No ', 'CCC', 'CDC', 'CEC', 'CED',
        # 'OpTime'], axis = 1)
        total_data2 = total_data2[['날짜', '시간', 'SOC', 'Max REGEN', 'Max POWER', 'Batt Current', 'Batt Volts', 'Temp 1', 'Max Cell V', 
                    'Min Cell V  ', 'Aux Batt Volts', 'Motor RPM 1', 'Motor RPM 2', 'SOH', 'Min Det']]
        total_data2.reset_index(drop=True, inplace=True)
        
        # plt.xlabel("Data point")
        # plt.ylabel("SOC")
        # plt.title('Total data of SOC')
        # plt.plot(total_data2['SOC'], 'b,')
        
        # plt.show()
        
        total_data2.to_csv(save_path + 'time_deleted_data ' + file_name)
        print(file_name + ' saved')

    