# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:47:21 2022

@author: gprua
"""
import numpy as np
import matplotlib.pylab as plt
N=200
xxRaw=np.random.randn(N)
yyRaw=np.random.randn(N)

plt.plot(xxRaw,yyRaw,"r.")

xx=np.c_[np.ones([N,1]),xxRaw]
yy=yyRaw.reshape([N,1])
wOLS=np.linalg.inv(xx.T.dot(xx)).dot(xx.T).dot(yy)

xxsample=np.linspace(-3,3,N)
xxsample_bias=np.c_[np.ones([N,1]),xxsample]
xxsample1=xxsample_bias.T
yysample=wOLS.T.dot(xxsample_bias.T)
plt.plot(xxsample.reshape([1,N]),yysample,"bo")

eta=0.2
n=1000
wGD=np.zeros([2,1])
for iteration in range(n):
    gradient=(-2/N)*(xx.T.dot(yy-xx.dot(wGD)))
    wGD=wGD-eta*gradient

d1=np.random.multivariate_normal(mean=[0,2],cov=[[2,-5],[-5,3]],size=N)
d2=np.random.multivariate_normal(mean=[8,6],cov=[[5,-3],[-3,8]],size=N)

f3=plt.figure()
ax3=plt.axes(projection='3d')
ax3.plot(d1[:,0],d1[:,1],0,'r.')
ax3.plot(d2[:,0],d1[:,1],1,'b.')


#xT(u-y)
x1=np.c_[np.ones([N,1]),d1]
x2=np.c_[np.ones([N,1]),d2]
x=np.r_[x1,x2]
y1=np.zeros([N,1])
y2=np.ones([N,1])
y=np.r_[y1,y2]

def sigmoid(xxx):
    return 1/(1+np.exp(-xxx))

wGD2=np.zeros([3,1])
for iteration in range(n):
    u=sigmoid(wGD2.T.dot(x.T)).T
    NLL=x.T.dot(u-y)
    wGD2=wGD2-eta*NLL

x1plot=np.linspace(-5,6,N)
x2plot=np.linspace(-5,6,N)
[x1plot,x2plot]=np.meshgrid(x1plot,x2plot)
yplot=sigmoid(wGD2[0]+wGD2[1]*x1plot+wGD2[2]*x2plot)
ax3.plot_surface(x1plot,x2plot,yplot,cmap='viridis')












