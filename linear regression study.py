# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 22:12:18 2022

@author: gprua
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dfLoad=pd.read_csv('https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LinearRegression.txt',sep="\s+")
xxRaw=np.array(dfLoad.values[:,0])
yyRaw=np.array(dfLoad.values[:,1])
plt.plot(xxRaw,yyRaw,"r.")

N=len(xxRaw)
xx=np.c_[np.ones([N,1]),xxRaw]#N*2
yy=yyRaw.reshape(N,1)#N*1

wOLS=np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)#2*1
xxPredict=np.linspace(0,2,num=N)#N*1
xxpredictPadding=np.c_[np.ones([N,1]),xxPredict]#N*2
yyPredict=wOLS.T.dot(xxpredictPadding.T)#연산을 위해 기존 식에 x를 xT로 대체
#1*N
plt.plot(xxPredict.reshape(1,100),yyPredict,"b.-")
#x와 y의 모양을 반대로 맞춰 주어야 한다.

eta=0.2
wGD=np.zeros([2,1])
n=200
for iteration in range(n):
    gradient=-(2/N)*(xx.T.dot(yy-xx.dot(wGD)))
    wGD=wGD-eta*gradient
