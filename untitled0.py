# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:28:26 2022

@author: gprua
"""

import numpy as np
import matplotlib.pylab as plt

N=200
# xxRaw=np.random.randn(N)
# yyRaw=np.random.randn(N)
# plt.plot(xxRaw,yyRaw,"r.")

# xxpadding=np.c_[np.ones([N,1]),xxRaw]
# yy=yyRaw.reshape([N,1])

# wOLS=np.linalg.inv(xxpadding.T.dot(xxpadding)).dot(xxpadding.T).dot(yy)
# xxsample=np.linspace(-5,5,N)
# xxsample_bias=np.c_[np.ones([N,1]),xxsample]
# yysample=wOLS.T.dot(xxsample_bias.T)
# plt.plot(xxsample.reshape([1,N]),yysample,'b.-')

# eta=0.2
# n=100
# wGD=np.zeros([2,1])
# for iteration in range(n):
#     gradient=(-2/n)*(xxpadding.T.dot(yy-xxpadding.dot(wGD)))
#     wGD=wGD-eta*gradient

d1=np.random.multivariate_normal(mean=[0,2],cov=[[2,-5],[-5,3]],size=N)
d2=np.random.multivariate_normal(mean=[8,6],cov=[[5,-3],[-3,8]],size=N)

f3=plt.figure()
ax3=plt.axes(projection='3d')
ax3.plot(d1[:,0],d1[:,1],0,'r.')
ax3.plot(d2[:,0],d2[:,1],1,'b.')

def sigmoid(x):
    return 1/(1+np.exp(-x))



#xT(u-y)
xx1=np.c_[np.ones([N,1]),d1]
xx2=np.c_[np.ones([N,1]),d2]
xx=np.r_[xx1,xx2]
yy1=np.zeros([N,1])
yy2=np.ones([N,1])
yy=np.r_[yy1,yy2]

eta=0.2
n=100
wGD=np.zeros([3,1])

for iteration in range(n):
    u=sigmoid(wGD.T.dot(xx.T)).T
    gradient=xx.T.dot(u-yy)
    wGD=wGD-eta*gradient

xxsample1=np.linspace(-5,15,N)
xxsample2=np.linspace(-5,15,N)
[xxsample1,xxsample2]=np.meshgrid(xxsample1,xxsample2)
yysample=sigmoid(wGD[0]+wGD[1]*xxsample1+wGD[2]*xxsample2)
ax3.plot_surface(xxsample1,xxsample2,yysample,cmap='viridis')













