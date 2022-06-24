# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 23:41:00 2022

@author: gprua
"""
#library 사용
from sklearn.datasets import load_iris
#sklearn library내부에 있는 load_iris라는 methrod를 가져온다
iris =load_iris()
#iris라는 변수에 load_iris dataset을 저장
irisDataInput =iris.data
#iris 내부의 data라는 set을 input에 저장

#knn 구현(150개중 90개는 trainning 60개는 test로 사용)
from sklearn.model_selection import train_test_split#위를 구현해주는 라이브러리
X= iris.data#input
y=iris.target#class의 정답
x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.4,random_state=42)
#test_size 150*0.4=60개로 테스트 하겟다. random_state은 trainning과 test를 어떻게 split하겠냐를 정해준다.

from sklearn.neighbors import KNeighborsClassifier#knn 알고리즘의 method
from sklearn import metrics#trainning과 test를 비교하여 성능(accuracy)을 percent로 알려준다

knn=KNeighborsClassifier(n_neighbors=5)#k=5
knn.fit(x_train,y_train)#대입

y_pred=knn.predict(x_test)#우리가 예상한값(class)
scores=metrics.accuracy_score(y_test, y_pred)#결과와 우리의 예상비

print(scores)
# from sklearn.datasets import load_breast_cancer
# bCancer=load_breast_cancer()