# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:15:25 2022

@author: gprua
"""

# 1. 데이터 개수 설정
# 2. 랜덤변수 설정
# 3. 평균,covariance 설정

import numpy as np
import matplotlib.pylab as plt

N=100
a=np.random.randn(N)
d1=np.random.multivariate_normal(mean=[0,2],cov=[[2,-5],[-5,3]],size=N)
d2=np.random.multivariate_normal(mean=[8,6],cov=[[5,-3],[-3,8]],size=N)

#4플랏 형성
#5인풋값으로 점 찍기
f3=plt.figure()
ax3=plt.axes(projection='3d')
ax3.plot(d1[:,0],d1[:,1],0,'r.')
ax3.plot(d2[:,0],d2[:,1],1,'b.')

#6 xT(u-y)구현
x1=np.c_[np.ones([N,1]),d1]
x2=np.c_[np.ones([N,1]),d2]
x=np.r_[x1,x2]
y1=np.zeros([N,1])
y2=np.ones([N,1])
y=np.r_[y1,y2]

def sigmoid(xx):
    return 1/(1+np.exp(-xx))

#7gradient descent 구현
eta=0.2
n_iteration=200
wGD=np.zeros([3,1])
for iteration in range(n_iteration):
    u=sigmoid(wGD.T.dot(x.T)).T
    gradient=x.T.dot(u-y)
    wGD=wGD-eta*gradient

#8우리가 정한 시그모이드 평면을 3d 그래프에 구현
x1sig=np.linspace(-5,15,100)
x2sig=np.linspace(-5,15,100)
[x1sig,x2sig]=np.meshgrid(x1sig,x2sig)
ysig=sigmoid(wGD[0]+wGD[1]*x1sig+wGD[2]*x2sig)
ax3.plot_surface(x1sig,x2sig,ysig,cmap='viridis')


