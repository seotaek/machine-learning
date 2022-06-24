# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 23:28:19 2022

@author: gprua
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m

plt.close("a11")
#database에는 data이름&index 추가 가능

dfLoad=pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample.txt", sep="\s+")
x=np.array(dfLoad["X"])#해당열을 가져온다
y=np.array(dfLoad["Y"])#dfLoad의 Y열을 가져온다.
"""dfLoad.iloc[]: 해당 행의 data를 가져온다"""
N=len(x)
numK=2
np.random.seed(3)#랜덤값 고정
k=np.round(np.random.rand(N))#round는 반올림

"""dataframe"""
#좋은이유: df.groupby("K")로 알아서 classification 가능
#pd.DataFrame(X)를 통해서 data를 dataframe화 가능
# npCluster=np.c_[x,y,k]
# dfCluster=pd.DataFrame(npCluster)
# dfCluster.columns=["X","Y","K"]#열의 이름을 0 1 2가 아닌 x,y,k로 바꿔준다
# dfGroup=dfCluster.groupby("K")

# for a in dfGroup: class가0인 애들의 matrix 1개, class가 1인애들의 matrix 1개출력
#     print(a)
# for (cluster,dataInCluster) in dfGroup: #말그대로
#     print(dataInCluster)

samples=np.array(dfLoad)
f1=plt.figure(1)#판 추가
ax1=f1.add_subplot(111)#축 추가,(2는행갯수,3은 열갯수, 번호에 맞는 판에 그래프 생성)
ax1.plot(x,y,'b.')

"""kmean clustering"""#23
[mx,sx]=[np.mean(x),np.std(x)]
[my,sy]=[np.mean(y),np.std(y)]
"""완전 이상한 값이 나오는 것은 피하기 위해 전체 평균과 표준편차 추출"""
z0=np.array([mx+sx,my+sy]).reshape(1,2)
z1=np.array([mx-sx,my-sy]).reshape(1,2)
z=np.r_[z0,z1]
ax1.plot(z[:,0],z[:,1],'r*',markersize='20')#32번째 줄이랑 같은 그래프안에 다른 색상으로 그래프 표시
"""z0,z1"""
"""=>z 초기화"""


k=np.zeros(N)

while(True):
    kOld=np.copy(k)#k라고 쓸경우 k가 업데이트 됨에 따라 kOld도 같이 업데이트 되어서 이렇게 사용
    """x와 y가 z0에 가까운지, z1에 가까운지 비교"""
    for i in np.arange(N):
        z0D =np.linalg.norm(samples[i,:]-z[0,:])
        z1D =np.linalg.norm(samples[i,:]-z[1,:])
        k[i]=z0D>z1D
    
    if(np.alltrue(kOld==k)):
        break
            # if(z0D<z1D):
            #     k[i]=0
            # else:
            #     k[i]=1  ->이런의미
        
    dfCluster=pd.DataFrame(np.c_[x,y,k])#groupby를 하기 위해 dataframe 화
    dfCluster.columns=["X","Y","K"]#열의 이름을 0 1 2가 아닌 x,y,k로 바꿔준다
    dfGroup=dfCluster.groupby("K")
    """k에 따라 classification"""
        # for(cluster,dataInCluster) in dfGroup:
        #     print(dataInCluster)
        

    for cluster in range(numK):
        z[cluster,:]=dfGroup.mean().iloc[cluster]

# f2=plt.figure(1)#판 추가
# ax2=f2.add_subplot(111)#축 추가,2는행갯수,3은 열갯수
# ax2.plot(x,y,'b.')

# for(cluster,dataInCluster) in dfGroup:
#     ax2.plot(dataInCluster.X,dataInCluster.Y,'.')

ax1.plot(z[:,0],z[:,1],'g*',markersize='20')


































