# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:46:01 2022

@author: gprua
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m
import scipy as sp
import scipy.stats
"""순서"""
# 1. data 가져오기
# 2. 파이, 시그마, u 초기화
# 3. responsibility 초기화

# while문
# 4. N0,N1을 normal distribution으로 표현
# 5. E-step, M-step 구현

# 6. group by를 위한 dataframe화
# 7. 그래프로 표현
plt.close("all")

dfLoad=pd.read_csv('https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample2.txt', sep='\s+')
samples=np.array(dfLoad)
numK=2
x=samples[:,0]
y=samples[:,1]
N=len(x)

"""초기값 설정"""
pi=np.ones(numK)*(1/numK)#파이
[mx, sx]=[np.mean(x),np.std(x)]
[my, sy]=[np.mean(y),np.std(y)]

u0=np.array([mx-sx,my+sy])#평균
u1=np.array([mx+sx,my-sy])
sigma0=np.array([[sx*sx/4,0 ],[0,sy*sy/4]])#covariance
sigma1=np.array([[sx*sx/4,0 ],[0,sy*sy/4]])

f1=plt.figure(1)
ax1=f1.add_subplot(111)
ax1.plot(x,y,'.b')
ax1.plot([u0[0],u1[0]],[u0[1],u1[1]],'*r',markersize=20)


"""gmm clustering"""
R=np.ones([N,numK])*(1/numK)#resposibility 초기화
j=0
while(True):
    j+=1
    N0=sp.stats.multivariate_normal.pdf(samples,u0,sigma0)
    #normal distribution
    N1=sp.stats.multivariate_normal.pdf(samples,u1,sigma1)
    
    #E-step(R-update)
    Rold=np.copy(R)#while문 빠져나가는 condition
    R=np.array([pi[0]*N0/(pi[0]*N0+pi[1]*N1), pi[1]*N1/(pi[0]*N0+pi[1]*N1)]).T

    if(np.linalg.norm(R-Rold)<N*numK*0.00000000001):#오차범위가 일정 범위 이내이면 while문 종료
        break

    #M-step(pi,sigma,u update)
    pi =np.ones(N).reshape(1,N).dot(R)/N
    pi =pi.reshape(2,)
    weightedSum =samples.T.dot(R)
    u0 =weightedSum[:,0]/sum(R[:,0])
    u1 =weightedSum[:,1]/sum(R[:,1])
    Sigma0 =samples.T.dot(np.multiply(R[:,0].reshape(N,1), samples))/sum(R[:,0]) -u0.reshape(2,1)*u0.reshape(2,1).T
    Sigma1 =samples.T.dot(np.multiply(R[:,1].reshape(N,1), samples))/sum(R[:,1]) -u1.reshape(2,1)*u1.reshape(2,1).T

print(j)
k=np.round(R[:,1])#hard clustering
dfCluster=pd.DataFrame(np.c_[x,y,k])
dfCluster.columns=["X","Y","K"]
dfGroup=dfCluster.groupby("K")

f2=plt.figure(2)
ax2=f2.add_subplot(111)
for(dfCluster,dataGroup) in dfGroup:
    ax2.plot(dataGroup.X,dataGroup.Y,".",label=dfCluster)
"""feature scaling(data standardization)해주는 이유"""
# gradient descent에 의해 w가 바뀔때 delta NLL이 scale에 따라 바뀌기 때문에 이 scale을 [0,1]로 고정시키기 위해 사용한다.



























