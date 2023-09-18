#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:30:04 2021

@author: leehungi
"""

import matplotlib.pyplot as plt #import plot
from mat2json import loadMat
from util import getBatteryCapacity, getChargingValues, getDischargingValues, getDataframe, series_to_supervised, rollingAverage

B0005 = loadMat('B0005.mat')
B0006 = loadMat('B0006.mat')
B0007 = loadMat('B0007.mat')
B0018 = loadMat('B0018.mat')

B0005_capacity = getBatteryCapacity(B0005)
B0006_capacity = getBatteryCapacity(B0006)
B0007_capacity = getBatteryCapacity(B0007)
B0018_capacity = getBatteryCapacity(B0018)

fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(B0005_capacity[0], B0005_capacity[1], color='blue', label='Battery-05')
ax.plot(B0006_capacity[0], B0006_capacity[1], color='green', label='Battery-06')
ax.plot(B0007_capacity[0], B0007_capacity[1], color='red', label='Battery-07')
ax.plot(B0018_capacity[0], B0018_capacity[1], color='purple', label='Battery-18')
ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()

charging_labels = ['Voltage_measured','Current_measured','Temperature_measured']

B0005_charging = getChargingValues(B0005, 0)
B0006_charging = getChargingValues(B0006, 0)
B0007_charging = getChargingValues(B0007, 0)
B0018_charging = getChargingValues(B0018, 0)

indx = 1
for label in charging_labels:
    fig, ax = plt.subplots(1, figsize=(20, 7))
    fig1, ax1 = plt.subplots(1, figsize=(20, 7))
    fig2, ax2 = plt.subplots(1, figsize=(20, 7))
    fig3, ax3=plt.subplots(1, figsize=(20, 7))

    ax.plot(B0005_charging[5], B0005_charging[indx], color='red' ,label="B0005")
    ax1.plot(B0006_charging[5], B0006_charging[indx], color='orange', label="B0006")
    ax2.plot(B0007_charging[5], B0007_charging[indx], color='blue' ,label="B0007")
    ax3.plot(B0018_charging[5], B0018_charging[indx], color='pink' ,label="B0018")

    ax.set(xlabel='Time in seconds', ylabel=label, title='Charging performance ')
    plt.legend()
    ax1.set(xlabel='Time in seconds', ylabel=label, title='Charging performance ')
    ax2.set(xlabel='Time in seconds', ylabel=label, title='Charging performance')
    ax3.set(xlabel='Time in seconds', ylabel=label, title='Charging performance')
    
    
    indx += 1    
    
B0005_discharging = getDischargingValues(B0005, 1)
B0006_discharging = getDischargingValues(B0006, 1)
B0007_discharging = getDischargingValues(B0007, 1)

indx = 1
for label in charging_labels:
    fig, ax = plt.subplots(1, figsize=(20, 7))
    fig1, ax1 = plt.subplots(1, figsize=(20, 7))
    fig2, ax2 = plt.subplots(1, figsize=(20, 7))
    
    ax.plot(B0005_discharging[5], B0005_discharging[indx], color='red' ,label="B0005")
    ax1.plot(B0006_discharging[5], B0006_discharging[indx], color='orange', label="B0006")
    ax2.plot(B0007_discharging[5], B0007_discharging[indx], color='blue' ,label="B0007")
    
    ax.set(xlabel='Time in seconds', ylabel=label, title='DisCharging performance ')
    plt.legend()
    ax1.set(xlabel='Time in seconds', ylabel=label, title='DisCharging performance ')
    ax2.set(xlabel='Time in seconds', ylabel=label, title='DisCharging performance')
    
    
    indx += 1    
    