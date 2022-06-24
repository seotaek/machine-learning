# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 23:57:45 2022

@author: gprua
"""

import numpy as np
import matplotlib.pylab as plt

N=200
xxRaw=np.random.randn(N)
yyRaw=np.random.randn(N)
#x,y의 랜덤 데이터 대입

plt.plot(xxRaw,yyRaw,"r.")
xx=np.c_[np.ones([N,1]),xxRaw]#x에 1을 padding
yy=yyRaw.reshape([N,1])

#w=(xTx)xTy구현
wOLS=np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)
xxsample=np.linspace(-3,3,N)#직선을 구하기 위해 인풋의 범위 만큼 지정
xxsample_bias=np.c_[np.ones([N,1]),xxsample] #1을 padding
yysample=wOLS.T.dot(xxsample_bias.T)
plt.plot(xxsample.reshape([1,N]),yysample,"bo")

#gradient descent
eta=0.2
wGD=np.zeros([2,1])
n=1000
for iteration in range(n):
    gradient=-(2/N)*(xx.T.dot(yy-xx.dot(wGD)))
    wGD=wGD-eta*gradient