# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:05:08 2022

@author: gprua
"""
import numpy as np
x=np.array([[1,3],[2,4]])
y=np.array([[1,6],[3,0]])
a=np.arange(9).reshape((3,3))

#1)np.dot(x,y)
print(np.dot(x,y))
#행렬과 행렬간의 곱 또는 서로 다른 차원의 곱을 할 때 사용

#2)np.diag
print(a)
print(np.diag(a))#행렬의 대각선 성분을 뽑을 때 사용
print(np.diag(a,k=1))#행렬의 대각선 오른쪽 위 대각선 성분을 뽑을 때 사용
print(np.diag(a,k=-1))#행렬의 대각선 왼쪽 아래의 대각선 성분을 뽑을 때 사용

#3)np.trace
print(np.trace(a))#행렬의 대각선 합 성분 추출
#만약 정사각행렬이 아닐경우
b=np.arange(1,9).reshape((2,4))
#[[1 2 3 4]
# [5 6 7 8]]
print(np.trace(b))#1+6
print(np.trace(b,1))#2+7
print(np.trace(b,2))#3+8
print(np.trace(b,3))#4의 대각선 성분은 없으므로 4이다

#4)np.linalg.det
b=np.array([[1,2],
            [3,4]])
print(np.linalg.det(b))#행렬의 determinant값 출력

#5)np.linalg.inv
b_inv=np.linalg.inv(b)
print(b_inv)# 역행렬 출력

#6)np.linalg.svd
c=np.arange(1,10).reshape(3,3)
print(c)
# svd함수를 사용하여 3개의 반환값(U,s,V)를 저장한다.

U, s, V = np.linalg.svd(c, full_matrices = True)

# s는 c의 고유값(eigenvalue) 리스트이다.

# svd를 이용하여 근사한(approximated) 결과를 원본과 비교하기 위해

# s를 유사대각행렬로 변환한다.

S = np.zeros(c.shape)

for i in range(len(s)):
    S[i][i] = s[i]

# 근사한 결과를 계산한다.
appA = np.dot(U, np.dot(S, V))
print(c-appA)
#0에 근접하므로 매우 유사한것을 확인할 수 있다.

#7)np.linalg.solve
# 4x+3y=23
# 3x+2y=16 #이 연립방정식의 해를 구할 때 사용
x=np.array([[4,3],[3,2]])
y=np.array([23,16])
print(np.linalg.solve(x,y))