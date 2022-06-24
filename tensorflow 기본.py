# -*- coding: utf-8 -*-
"""
Created on Fri May 20 21:12:05 2022

@author: gprua
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_H_vs_W.txt", sep ="\s+")
HeightRaw = np.array(dfLoad["Height"]).reshape(-1,1) #벡터를 행렬로(100*1)
WeightRaw=np.array(dfLoad["Weight"]).reshape(-1,1)

#data standardization
scaler = StandardScaler()
scaler.fit(HeightRaw)
Height_std = scaler.transform(HeightRaw)

nData = np.shape(HeightRaw)[0]#data뽑기
X_np =np.c_[np.ones(nData),Height_std]
y_np=WeightRaw

X=tf.constant(X_np,dtype=tf.float32)#변할일 없는 변수를 constant로 선언
#training되는건 variable
y=tf.constant(y_np,dtype=tf.float32)
theta=tf.Variable(tf.random.uniform([2,1],-1,1),dtype=tf.float32)#random한 initial value와 type 선언

learning_rate=0.1
n_epoch=100#전체 trainingset을 100번 돌려서 weight 업데이트
for i in np.arange(n_epoch):
    with tf.GradientTape() as g:#gradient term을 몰라도 알아서 업데이트 되는 형태
        y_pred=tf.matmul(X,theta)#y_pred=w*x
        error=y-y_pred
        mse=tf.reduce_mean(tf.square(error))#(y-y_pred)제곱
        #theta와 mse의 관계식을 나타내는 구간
    gradients=g.gradient(mse,[theta])#mse를 theta에 대해 미분
    theta.assign(theta-learning_rate*gradients[0].numpy())
    if(n_epoch%10==0):
        print(theta)

w1_std=theta.numpy()[1]
w0_std=theta.numpy()[0]

xPlt_std=np.linspace(-2,2,100)
f2,ax2=plt.subplots()
ax2.plot(Height_std, WeightRaw, "r.")
ax2.plot(xPlt_std, w1_std*xPlt_std+w0_std,"g--")#feature scaling 후의 그래프

w1=w1_std/np.sqrt(scaler.var_)
w0=w0_std-scaler.mean_*w1_std/np.sqrt(scaler.var_)
#standardization한 결과를 다시 원래의 데이터 크기로 돌려주는 과정

xPltRaw=np.linspace(min(HeightRaw),max(HeightRaw),100)
f3,ax3=plt.subplots()
ax3.plot(HeightRaw, WeightRaw,"r.")
ax3.plot(xPltRaw,w1*xPltRaw+w0,"g--")#feature scaling 이전의 데이터 크기로 그린 그래프

























