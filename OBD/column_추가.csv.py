# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 11:43:50 2022

@author: tako
"""
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import numpy as np
import csv


path = "C:\\Users\\tako\\Desktop\\EV log\\전체데이터" # 하나 폴더로 생성
save_path = "C:\\Users\\tako\\Desktop\\EV log\\column_data\\"
file_list = os.listdir(path)
columns = ['날짜', '시간', 'SOC', 'Max REGEN', 'Max POWER', 'BMS Relay', 'Batt Current', 'Batt Volts', 'Temp 1', 'Temp 2', 'Temp 3', 'Temp 4', 'Temp 5', 'Max Cell V', 
           'Max Cell V No    ', 'Min Cell V  ', 'Min Cell V No ', 'Aux Batt Volts', 'CCC', 'CDC', 'CEC', 'CED', 'OpTime', 'Motor RPM 1', 'Motor RPM 2', 'SOH', 'Max Det Cell No', 
           'Min Det', 'Min Det Cell No', 'Cooling water T']
			

for file_name in file_list:
    if '쏘울' in file_name:
        f=open(path +'/'+file_name, encoding="cp949")
        reader = csv.reader(f)
        '''리스트 리셋'''
        csv_list=[] 
        for l in reader:
            csv_list.append(l)
        f.close()
        total_data = pd.DataFrame(csv_list)
        total_data = total_data.drop([0,1])
                
        temp_column = columns
        if len(total_data.columns) != len(columns):
            if len(total_data.columns) - len(columns) < 0:
                temp_column = temp_column[:len(total_data.columns) - len(columns)]
            else:
                for append_num in range(len(total_data.columns) - len(columns)):
                    if len(total_data.columns) - len(columns) >0:
                        temp_column.append('v' + str(91+append_num))
                ##마지막줄 왜 v+str()넣는거지????
        total_data.columns = temp_column
        total_data['SOC'] = total_data['SOC'].astype('float')
        mask = total_data['SOC'] > 0
        total_data = total_data.loc[mask, :]
        
        print(file_name)
        total_data.to_csv(save_path + 'with_column ' + file_name)
        print('save' + file_name + ' with_column')
       
    else:

    
    
        f=open(path +'/'+file_name, encoding="cp949")
        reader = csv.reader(f)
        '''리스트 리셋'''
        csv_list=[] 
        for l in reader:
            csv_list.append(l)
        f.close()
        total_data = pd.DataFrame(csv_list)
        
        temp_column = columns
        if len(total_data.columns) != len(columns):
            if len(total_data.columns) - len(columns) < 0:
                temp_column = temp_column[:len(total_data.columns) - len(columns)]
            else:
                for append_num in range(len(total_data.columns) - len(columns)):
                    if len(total_data.columns) - len(columns) >0:
                        temp_column.append('v' + str(91+append_num))
                
        total_data.columns = temp_column
        total_data['SOC'] = total_data['SOC'].astype('float')
        mask = total_data['SOC'] > 0
        total_data= total_data.loc[mask, :]
        
        print(file_name)
        total_data.to_csv(save_path + 'with_column ' + file_name)
        print('save' + file_name + ' with_column')