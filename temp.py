# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dfLoad=pd.read_csv('https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LinearRegression.txt',sep="\s+")
xxRaw=np.array(dfLoad.values[:,0])
yyRaw=np.array(dfLoad.values[:,1])
plt.plot(xxRaw,yyRaw,"r.")

N=len(xxRaw)
xx_bias=np.c_[np.ones([100,1]),xxRaw]
yy=yyRaw.reshape(N,1)

wOLS=np.linalg.inv(xx_bias.T.dot(xx_bias)).dot(xx_bias.T).dot(yy)
x_sample=np.linspace(0,2,101)#0~2의 행렬 설정
x_sample_bias=np.c_[np.ones([101,1]),x_sample]#1행에 성분이 1인 행 추가
y_bias=wOLS.T.dot(x_sample_bias.T)#행렬끼리 곱하기 위해서 x를 transpose
plt.plot(x_sample.reshape(1,101),y_bias,"b.-")
eta=0.1
n_iteration=1000
wGD=np.zeros([2,1])
print(wGD)
for iteration in range(n_iteration):
    gradients=-(2/N)*(xx_bias.T.dot(yy-xx_bias.dot(wGD)))
    wGD=wGD-eta*gradients
print(wGD)