import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from mat2json import loadMat
from util import getBatteryCapacity, getChargingValues, getDischargingValues, getDataframe, series_to_supervised, rollingAverage, getSOH, build_dataset




#Ambient temp 24
B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')



dfB0005 = getDataframe(B0005)

B0005_capacity = getBatteryCapacity(B0005)
B0006_capacity = getBatteryCapacity(B0006)
B0007_capacity = getBatteryCapacity(B0007)
B0018_capacity = getBatteryCapacity(B0018)

sns.residplot(dfB0005['cycle'], dfB0005['capacity'])


df = pd.DataFrame({"Voltage_load":B0005[1]['data']['Voltage_load'], "Time":B0005[1]['data']['Time']})
idx = df[df["Voltage_load"] == 0.000].index
df.drop(idx, inplace = True)
len(df)
# B0005_SOH = getSOH(B0005_capacity)
# B0005_SOH = {'SOH' : B0005_SOH}
# B0005_SOH = pd.DataFrame(B0005_SOH)

dfB0005 = getDataframe(B0005)
dfB0006 = getDataframe(B0006)
dfB0007 = getDataframe(B0007)
dfB0018 = getDataframe(B0018)

X = dfB0006.iloc[:,[2, 3,4, 5]]
Y = dfB0006.iloc[:,[6]]

# min_max_scaler = MinMaxScaler()
# Nor = min_max_scaler.fit(x)
# X_Nor = Nor.transform(x)
# Data_X, Data_Y = build_dataset(X_Nor, Y, 2)


X = dfB0006['Time']
Y = dfB0006['SOH']
# A = dfB0007['Time']
# B = dfB0007['SOH']

# ratio = 40
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
# lst_x, lst_y = rollingAverage(X_train, y_train)
# d = {'X_train':X_train.values,'y_train':y_train.values}
# d = pd.DataFrame(d)
# d = d[~d['X_train'].isin(lst_x)]
# X_train = d['X_train']
# y_train = d['y_train']
# X_train = X_train.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)
# best_svr = SVR(C=20, epsilon=0.0001, gamma=0.0001, cache_size=200,
#   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# best_svr.fit(X_train,y_train)
# if ratio == 40:
#     y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))


ratios = [40, 30, 20, 10]

    
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
    lst_x, lst_y = rollingAverage(X_train, y_train)
    d = {'X_train':X_train.values,'y_train':y_train.values}
    d = pd.DataFrame(d)
    d = d[~d['X_train'].isin(lst_x)]
    X_train = d['X_train']
    y_train = d['y_train']
    X_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    best_svr = SVR(C=20, epsilon=0.0001, gamma=0.0001, cache_size=200,
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    best_svr.fit(X_train,y_train)
    if ratio == 40:
        y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))
    elif ratio == 30:
        y_pred_30 = best_svr.predict(X.values.reshape(-1, 1))
    elif ratio == 20:
        y_pred_20 = best_svr.predict(X.values.reshape(-1, 1))
    elif ratio == 10:
        y_pred_10 = best_svr.predict(X.values.reshape(-1, 1))
        
    
fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(Y, color='black', label='Battery Capacity')
ax.plot(y_pred_40, color='red', label='Prediction with train size of 60%')
ax.plot(y_pred_30, color='blue', label='Prediction with train size of 70%')
ax.plot(y_pred_20, color='green', label='Prediction with train size of 80%')
ax.plot(y_pred_10, color='yellow', label='Prediction with train size of 90%')

ax.set(xlabel='Time in seconds', ylabel='capacity', title='Model performance for Battery 05')
ax.legend()


# X = dfB0007['cycle']
# Y = dfB0007['capacity']


# fig, ax = plt.subplots(1, figsize=(12, 8))


# ax.scatter(X, Y, color='green', label='Battery')


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)


# lst_x, lst_y = rollingAverage(X_train, y_train)
# d = {'X_train':X_train.values,'y_train':y_train.values}
# d = pd.DataFrame(d)
# d = d[~d['X_train'].isin(lst_x)]
# X_train = d['X_train']
# y_train = d['y_train']

# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.scatter(X_train, y_train, color='green', label='Battery capacity data')
# ax.scatter(lst_x, lst_y, color='red', label='Outliers')
# ax.legend()

# X_train = X_train.values.reshape(-1, 1)
# y_train = y_train.values.reshape(-1, 1)

# best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
#   kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

# best_svr.fit(X_train,y_train)

# y_pred = best_svr.predict(X.values.reshape(-1, 1))

# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.plot(X, Y, color='green', label='Battery capacity data')
# ax.plot(X, y_pred, color='red', label='Fitted model')
# ax.set(xlabel='Time in seconds', ylabel='capacity', title='Discharging performance at 43°C')
# ax.legend()



# X = dfB0005['max_discharge_temp']
# Y = B0005_SOH
# ratios = [40, 30, 20, 10]
# for ratio in ratios:
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
#     lst_x, lst_y = rollingAverage(X_train, y_train)
#     d = {'X_train':X_train.values,'y_train':y_train.values}
#     d = pd.DataFrame(d)
#     d = d[~d['X_train'].isin(lst_x)]
#     X_train = d['X_train']
#     y_train = d['y_train']
#     X_train = X_train.values.reshape(-1, 1)
#     y_train = y_train.values.reshape(-1, 1)
#     best_svr = SVR(C=20, epsilon=0.0001, gamma=0.0001, cache_size=200,
#       kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#     best_svr.fit(X_train,y_train)
#     if ratio == 40:
#         y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 30:
#         y_pred_30 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 20:
#         y_pred_20 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 10:
#         y_pred_10 = best_svr.predict(X.values.reshape(-1, 1))
        
    
# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.plot(Y, color='black', label='Battery Capacity')
# ax.plot(y_pred_40, color='red', label='Prediction with train size of 60%')
# ax.plot(y_pred_30, color='blue', label='Prediction with train size of 70%')
# ax.plot(y_pred_20, color='green', label='Prediction with train size of 80%')
# ax.plot(y_pred_10, color='yellow', label='Prediction with train size of 90%')

# ax.set(xlabel='Time in seconds', ylabel='capacity', title='Model performance for Battery 05')
# ax.legend()





# X = dfB0006['max_discharge_temp']
# Y = dfB0006['capacity']
# ratios = [40, 30, 20, 10]
# for ratio in ratios:
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
#     lst_x, lst_y = rollingAverage(X_train, y_train)
#     d = {'X_train':X_train.values,'y_train':y_train.values}
#     d = pd.DataFrame(d)
#     d = d[~d['X_train'].isin(lst_x)]
#     X_train = d['X_train']
#     y_train = d['y_train']
#     X_train = X_train.values.reshape(-1, 1)
#     y_train = y_train.values.reshape(-1, 1)
#     best_svr = SVR(C=10, epsilon=0.0001, gamma=0.0001, cache_size=200,
#       kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#     best_svr.fit(X_train,y_train)
#     if ratio == 40:
#         y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 30:
#         y_pred_30 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 20:
#         y_pred_20 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 10:
#         y_pred_10 = best_svr.predict(X.values.reshape(-1, 1))
        
    
# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.plot(Y, color='black', label='Battery Capacity')
# ax.plot(y_pred_40, color='red', label='Prediction with train size of 60%')
# ax.plot(y_pred_30, color='blue', label='Prediction with train size of 70%')
# ax.plot(y_pred_20, color='green', label='Prediction with train size of 80%')
# ax.plot(y_pred_10, color='yellow', label='Prediction with train size of 90%')

# ax.set(xlabel='Time in seconds', ylabel='capacity', title='Model performance for Battery 06')
# ax.legend()


# X = dfB0007['max_discharge_temp']
# Y = dfB0007['capacity']
# ratios = [40, 30, 20, 10]
# for ratio in ratios:
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
#     lst_x, lst_y = rollingAverage(X_train, y_train)
#     d = {'X_train':X_train.values,'y_train':y_train.values}
#     d = pd.DataFrame(d)
#     d = d[~d['X_train'].isin(lst_x)]
#     X_train = d['X_train']
#     y_train = d['y_train']
#     X_train = X_train.values.reshape(-1, 1)
#     y_train = y_train.values.reshape(-1, 1)
#     best_svr = SVR(C=10, epsilon=0.0001, gamma=0.0001, cache_size=200,
#       kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#     best_svr.fit(X_train,y_train)
#     if ratio == 40:
#         y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 30:
#         y_pred_30 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 20:
#         y_pred_20 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 10:
#         y_pred_10 = best_svr.predict(X.values.reshape(-1, 1))
        
    
# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.plot(Y, color='black', label='Battery Capacity')
# ax.plot(y_pred_40, color='red', label='Prediction with train size of 60%')
# ax.plot(y_pred_30, color='blue', label='Prediction with train size of 70%')
# ax.plot(y_pred_20, color='green', label='Prediction with train size of 80%')
# ax.plot(y_pred_10, color='yellow', label='Prediction with train size of 90%')

# ax.set(xlabel='Time in seconds', ylabel='capacity', title='Model performance for Battery 07')
# ax.legend()



# X = dfB0018['max_discharge_temp']
# Y = dfB0018['capacity']
# ratios = [40, 30, 20, 10]
# for ratio in ratios:
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=False)
#     lst_x, lst_y = rollingAverage(X_train, y_train)
#     d = {'X_train':X_train.values,'y_train':y_train.values}
#     d = pd.DataFrame(d)
#     d = d[~d['X_train'].isin(lst_x)]
#     X_train = d['X_train']
#     y_train = d['y_train']
#     X_train = X_train.values.reshape(-1, 1)
#     y_train = y_train.values.reshape(-1, 1)
#     best_svr = SVR(C=20, epsilon=0.0001, gamma=0.00001, cache_size=200,
#       kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#     best_svr.fit(X_train,y_train)
#     if ratio == 40:
#         y_pred_40 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 30:
#         y_pred_30 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 20:
#         y_pred_20 = best_svr.predict(X.values.reshape(-1, 1))
#     elif ratio == 10:
#         y_pred_10 = best_svr.predict(X.values.reshape(-1, 1))
        
    
# fig, ax = plt.subplots(1, figsize=(12, 8))

# ax.plot(Y, color='black', label='Battery Capacity')
# ax.plot(y_pred_40, color='red', label='Prediction with train size of 60%')
# ax.plot(y_pred_30, color='blue', label='Prediction with train size of 70%')
# ax.plot(y_pred_20, color='green', label='Prediction with train size of 80%')
# ax.plot(y_pred_10, color='yellow', label='Prediction with train size of 90%')

# ax.set(xlabel='Time in seconds', ylabel='capacity', title='Model performance for Battery 18')
# ax.legend()
# plt.show()