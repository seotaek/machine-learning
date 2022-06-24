# -*- coding: utf-8 -*-
"""
Written by Hanwool Jeong at Home
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_H_vs_W.txt"
                     , sep ="\s+")
HeightRaw = np.array(dfLoad["Height"]).reshape(-1,1) #Vector to matrix
WeightRaw = np.array(dfLoad["Weight"]).reshape(-1,1)

scaler = StandardScaler()
scaler.fit(HeightRaw)
Height_std = scaler.transform(HeightRaw)

nData = np.shape(HeightRaw)[0]
learning_rate = 0.1
nEpochs = 100

X_np = np.c_[np.ones(nData), Height_std]
y_np = WeightRaw

X = tf.constant(X_np, dtype=tf.float32)
y = tf.constant(y_np, dtype=tf.float32)
theta = tf.Variable(tf.random.uniform([2,1], -1, 1), dtype=tf.float32) 

for i in np.arange(nEpochs):
    with tf.GradientTape() as g:
        y_pred = tf.matmul(X, theta)
        error = y - y_pred #nData x 1
        mse = tf.reduce_mean(tf.square(error))
    gradients = g.gradient(mse, [theta])
    theta.assign(theta - learning_rate * gradients[0].numpy())
    print(theta)

w1_std = theta.read_value().numpy()[1]
w0_std = theta.read_value().numpy()[0]

xPlt_std = np.linspace(-2, 2, 100)
f2, ax2 = plt.subplots()
ax2.plot(Height_std, WeightRaw, "r.")
ax2.plot(xPlt_std, w1_std*xPlt_std+w0_std, "g--")

w1 = w1_std/np.sqrt(scaler.var_)
w0 = w0_std - scaler.mean_*w1_std/np.sqrt(scaler.var_)

xPltRaw = np.linspace(min(HeightRaw), max(HeightRaw), 100)
f3, ax3 = plt.subplots()
ax3.plot(HeightRaw, WeightRaw, "r.")
ax3.plot(xPltRaw, w1*xPltRaw+w0, "g--")