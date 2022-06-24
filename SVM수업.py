# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:45:37 2022

@author: gprua
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC

#hyperparameter(standardization)
# X= np.array([
#     [2,-3],
#     [4,1],
#     [0, -2],
#     [10,3]])

# scaler=StandardScaler()
# scaler.fit(X)
# X_std=scaler.transform(X)#우리가 원하는 가우시안form으로 바꿔준다
"""순서"""
# 1. 데이터 입력
# 2. standardization
# 3. dataframe 화
# 4. svm 실행
# 5. margin 을 주어 x0,x1설정
# 6. meshgrid

iris=datasets.load_iris()

X=iris["data"][0:100,(2,3)]#design matrix(petal length, width)
Y=iris["target"][0:100]#classify된 output
f1,ax1=plt.subplots()
ax1.plot(X[:,0],X[:,1],'* ')

scaler=StandardScaler()#변수선언(object생성)
scaler.fit(X)#평균과 표준편차가 정해진다
X_std=scaler.transform(X)#우리가 원하는 가우시안form으로 바꿔준다

f2,ax2=plt.subplots()
ax2.plot(X_std[:,0],X_std[:,1],'* ')#figure1에 비해 scaling만 일어났다.

df_clf= pd.DataFrame(np.c_[X_std,Y])
df_clf.columns=["petalLength","petalWidth","target"]
df_clf_group =df_clf.groupby("target")
f3,ax3=plt.subplots()
for target, group in df_clf_group:
    ax3.plot(group.petalLength, group.petalWidth,'*',label="target")



svm_clf=SVC(C=0.01, kernel="linear")#kernel function 안 쓴 그대로의 svm
svm_clf.fit(X_std,Y)
#이 두줄로 svm이 끝
#predict는 말그대로 예측만 해준다, decision function을 이용하여 x가 h(x)에 들어간값을 구해야한다

[x0Min,x0Max]=[min(X_std[:,0])-0.1,max(X_std[:,0])+0.1]
[x1Min,x1Max]=[min(X_std[:,1])-0.1,max(X_std[:,1])+0.1]#margin을 주어 x0와 x1 설정
delta=0.01
[x0Plt,x1Plt]=np.meshgrid(np.arange(x0Min, x0Max,delta),np.arange(x1Min,x1Max,delta))#1. meshgrid
h=svm_clf.decision_function(np.c_[x0Plt.ravel(),x1Plt.ravel()])#2. ravel+ 3. concatination
h=h.reshape(x0Plt.shape)
CS=ax3.contour(x0Plt,x1Plt,h,cmap=plt.cm.twilight)
ax3.clabel(CS)#등고선의 수치 표현

#keneltrick:각자찾아서 해보고 noise, factor
#kernel=rbf-> hyper parameter: gamma,
#poly hyperparameter:degree,gamma,coef0

"""contour"""
#ax3.contour(x0Plt,x1Plt,h)
# x0Plt=[1,2,3
#        1,2,3]
# x1Plt=[0.3,0.3,0.3
#        0.6,0.6,0.6]
# h=[h(1,0.3),h(2,0.3),h(3,0.3)
#    h(1,0.6),h(2,0.6),h(3,0.6)]=>meshgrid를 사용하여 이러한 모양으로 생긴다

#1. meshgrid
# 2. x0Plt와 x1Plt를 각각 ravel 시킨다
#1차원배열=> 다차원 배열: reshape
#다차원 배열=> 1차원배열: ravel
# 3. concatination(np.c, np.r)
































