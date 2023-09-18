#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:36:59 2021

@author: leehungi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mat2json import loadMat
from util import getBatteryCapacity, getChargingValues, getDischargingValues, getDataframe, series_to_supervised, rollingAverage, getSOH
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


#Ambient temp 24
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')


# B0005_SOH = getSOH(B0005_capacity)
# B0005_SOH = {'SOH' : B0005_SOH}
# B0005_SOH = pd.DataFrame(B0005_SOH)

dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
dfB0007 = getDataframe(B0007)
dfB0018 = getDataframe(B0018)

X = dfB0006.iloc[:,[2, 4]]
Y = dfB0006.iloc[:,[6]]


x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
mlr.fit(x_train, y_train) 

y_predict = mlr.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Actual SOH")
plt.ylabel("Predicted SOH")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()

print(mlr.coef_)
accuracy = mlr.score(x_test, y_test)

# 상수항 추가
x_data2_ = sm.add_constant(X , has_constant = "add")

# 회귀모델 적합
multi_model2 = sm.OLS(Y, x_data2_)
fitted_multi_model2 = multi_model2.fit()


# 결과 출력
fitted_multi_model2.summary()

# #Ambient temp 24
# B0005 = loadMat('B0005.mat')
# B0006 = loadMat('B0006.mat')
# B0007 = loadMat('B0007.mat')
# B0018 = loadMat('B0018.mat')

# dfB0005 = getDataframe(B0005)



# B0005_capacity = getBatteryCapacity(B0005)
# B0006_capacity = getBatteryCapacity(B0006)
# B0007_capacity = getBatteryCapacity(B0007)
# B0018_capacity = getBatteryCapacity(B0018)

# getMaxChargeTemp(B0005)


# B0005_SOH = findSOH(B0005_capacity)

# B0005_charging = []
# for i in range(len(B0005)):
#     if B0005[i]['cycle'] == 'charge':
#         B0005_charging.append(getChargingValues(B0005, i))
        
    
        
# # B0005_discharging = []
# # for i in range(len(B0005)):
# #     if B0005[i]['cycle'] == 'discharge':
# #         B0005_discharging.append(getDischargingValues(B0005, i))
        
    
    
# B0005_charging = getChargingValues(B0005, 0)
# B0006_charging = getChargingValues(B0006, 0)
# B0007_charging = getChargingValues(B0007, 0)
# B0018_charging = getChargingValues(B0018, 0)

# B0005_discharging = getDischargingValues(B0005, 1)
# B0006_discharging = getDischargingValues(B0006, 1)
# B0007_discharging = getDischargingValues(B0007, 1)
# B0018_discharging = getDischargingValues(B0018, 2)

# labellist = list(B0005[0]['data'].keys())

# # with pd.ExcelWriter('charge.xlsx') as writer:
# #     for i in range(len(B0005_charging)):
# #         dfcharge = pd.DataFrame(B0005_charging[i][1:], labellist)
# #         charging = dfcharge.to_excel(writer, sheet_name='cycle' + str(i))
    

# # with pd.ExcelWriter('discharge.xlsx') as writer:
# #     for i in range(len(B0005_discharging)):
# #         dfdischarge = pd.DataFrame(B0005_discharging[i][1:], labellist)
# #         discharging = dfdischarge.to_excel(writer, sheet_name='cycle' + str(i))
        
# # df.to_excel('B0005.xlsx')

# charge_data = pd.read_excel('charge_data.xlsx', sheet_name=None, index_col= 0)

# discharge_data = pd.read_excel('discharge_data.xlsx', sheet_name=None, index_col= 0)




# # charge_data = charge_data.transpose()
# # discharge_data = discharge_data.transpose()

# # discharge_data.set_index(['time'])

# dfB0005 = getDataframe(B0005)
# dfB0006 = getDataframe(B0006)
# dfB0007 = getDataframe(B0007)
# dfB0018 = getDataframe(B0018)


# volt = []
# for i in discharge_data.keys():
#     volt.append(discharge_data[i].iloc[0])

# current_battery = []
# for i in discharge_data.keys():
#     current_battery.append(discharge_data[i].iloc[1])


# temp =[]
# for i in discharge_data.keys():
#     temp.append(discharge_data[i].iloc[2])

# current_load = []
# for i in discharge_data.keys():
#     current_load.append(discharge_data[i].iloc[3])

# voltage_load = []
# for i in discharge_data.keys():
#     voltage_load.append(discharge_data[i].iloc[4])

# t = []
# for i in discharge_data.keys():
#     t.append(discharge_data[i].iloc[4])


# x = timemin

# y = dfB0018['capacity']

# df = {}
# lisy = list(y)
# for i in range(len(x)):
#     df[x[i]] = lisy



# df = pd.DataFrame(df)


