# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:39:42 2022

@author: gprua
"""

import numpy as np
import matplotlib.pylab as plt

plt.close("all")

Ndata=1000
A=np.random.randn(Ndata)#Ndata만큼의 랜덤 변수 만들기
f1=plt.figure()#그림 출력 배경
ax1=plt.axes()#x축y축 설정
ax1.hist(A,bins=10)#bins는 가로축 구간의 개수(눈금 갯수)
#히스토그램을 만드는 모듈

#2D 가우시안 랜덤 variable 설정
f2=plt.figure()#그림 출력 배경
ax2=plt.axes()#x축y축 설정
d1=np.random.multivariate_normal(mean=[0,2],cov=[[2,-5],[-5,3]], size=Ndata)
#평균, covariance 설정
d2=np.random.multivariate_normal(mean=[8,6],cov=[[5,-3],[-3,8]], size=Ndata)
plt.scatter(d1[:,0],d1[:,1],c="b")#x축과 y축에 모두 점으로 찍기위해 설정
plt.scatter(d2[:,0],d2[:,1],c="r")

#3d plot
f3=plt.figure()#그림 출력 배경
ax3=plt.axes(projection='3d')#3dplot 설정, 빈칸이면 기존 2D로 기본값으로 가진다
ax3.plot(d1[:,0],d1[:,1],0,'b.')#0으로 claasify
ax3.plot(d1[:,0],d1[:,1],1,'r.')#1로 classify

#Xt(u-y)구현
N=Ndata
X1=np.c_[np.ones([N,1]),d1]#1짜리를 d1 붙이기
X2=np.c_[np.ones([N,1]),d2]
X=np.r_[X1,X2]#x1과 x2를 밑으로 붙이기
y1=np.zeros([N,1])
y2=np.ones([N,1])
y=np.r_[y1,y2]

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

eta=0.1
n_iterations=100
wGD=np.zeros([3,1])
wGDbuffer=np.zeros([3,n_iterations+1])

for iteration in range(n_iterations):
    mu= sigmoid(wGD.T.dot(X.T)).T
    gradients=X.T.dot(mu-y)
    wGD=wGD-eta*gradients
    wGDbuffer[:,iteration+1]=[wGD[0],wGD[1],wGD[2]]#확인

# x1sig=np.linspace(-2,2,100)#1차원 좌표
# x2sig=np.linspace(-2,2,100)
# [x1Sig,x2Sig]=np.meshgrid(x1sig,x2sig)#2차원 좌표로 변환
# ysig=sigmoid(x1Sig+x2Sig)

# f5=plt.figure()
# ax5=plt.axes(projection ='3d')
# ax5.plot_surface(x1Sig,x2Sig,ysig,cmap='viridis')

x1sig=np.linspace(-5,10,100)#1차원 좌표
x2sig=np.linspace(-5,10,100)
[x1Sig,x2Sig]=np.meshgrid(x1sig,x2sig)#x1과 x의 좌표를 2차원 좌표로 변환
ysig=sigmoid(wGD[1]*x1Sig+wGD[2]*x2Sig+wGD[0])
ax3.plot_surface(x1Sig,x2Sig,ysig,cmap='viridis')

