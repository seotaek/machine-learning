# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:59:24 2022

@author: gprua
"""
#MLP구현

"""순서"""
# 1. data 가져오기
# 2. feature scaling
# 3. one hot 10가지 class의 vector를 2가지 binary matrics 로
# 4. 하나의 neuron layer 를 정의 후 deepMLP 정의
# 5. 필요한 변수 선어(n_epoch, batchsize, nbatch, optimizer)
# 6. w와 b 초기화
# 7. deep MLP
# 8. accuracy 확인

#손글씨의 숫자를 판단(dataset(mnist))
#0 1 2 3 4 5 6 7 8 9
#0,1의 신호로 구분
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler#data standardization
plt.close("all")

#training할 data
(X_train, y_train),(X_test, y_test)=tf.keras.datasets.mnist.load_data()
nTrain=X_train.shape[0]#data 60000개

X_train_std,X_test_std=X_train/255.0,X_test/255.0 #feature scaling
X_train_std=X_train_std.astype("float32")

y_train_onehot=tf.keras.utils.to_categorical(y_train,10)
y_test_onehot=tf.keras.utils.to_categorical(y_test,10)
#원래는 10가지로 class의 vector를 2가지 binary class matrics로 바꿔준다

def neuron_layer(X,W,b,activation=None):#하나의 neuron layer 정의
    z=tf.matmul(X,W)+b
    if activation is None:
        return z
    else:
        return activation(z)


def my_dMLP(X_flatten,W,b):#neuron layer를 여러개 불러내서 deep MLP 구성
    hidden1=neuron_layer(X_flatten,W[0],b[0],activation=tf.nn.sigmoid)
    hidden2=neuron_layer(hidden1,W[1],b[1],activation=tf.nn.sigmoid)
    logits=neuron_layer(hidden2,W[2],b[2],activation=None)#output
    y_pred=tf.nn.softmax(logits)
    return y_pred

n_epoch=40#60000개의 data를 다 사용할때 n_epoch= 1
batchsize=200#60000개의 data중 200개를 사용할때 마다 w가 update
nBatch=int(nTrain/batchsize)#300번
opt=tf.keras.optimizers.SGD(learning_rate=0.01)#optimizer 선언
#SGD는 우리가 기본적으로 아는 gradient식

n_inputs =np.array([28*28, 256, 128])
n_nodes =np.array([256, 128, 10])
n_layer =3
W, b ={}, {}

for layer in range(n_layer):#w와b initialize
    stddev =2/np.sqrt(n_inputs[layer] +n_nodes[layer])
    W_init =tf.random.truncated_normal((n_inputs[layer], n_nodes[layer]), stddev =stddev)
    W[layer] =tf.Variable(W_init)
    b[layer] =tf.Variable(tf.zeros([n_nodes[layer]]))


for epoch in range(n_epoch):
    idxShuffle=np.random.permutation(X_train.shape[0])#random하게 data를shuffle 해준다
    for idxSet in range(nBatch):
        X_batch =X_train_std[idxShuffle[idxSet*batchsize:(idxSet+1)*batchsize], :]# 앞에서부터 batchsize만큼 data를 뽑아준다.
        X_batch_tensor =tf.convert_to_tensor(X_batch.reshape(-1, 28*28))#tensor화(28*28의 모양으로 되어 있었던 data를 784개의 data로 일렬로 나타낸다)
        y_batch =y_train_onehot[idxShuffle[idxSet*batchsize:(idxSet+1)*batchsize], :]#x_batch에 맞게 y_batch를 구축
        with tf.GradientTape() as tape:
            y_pred =my_dMLP(X_batch_tensor, W, b)#y predict를 b와W의 관계식으로 표현
            loss =tf.reduce_mean(tf.keras.losses.MSE(y_batch, y_pred))#MSE를 vector모양이 아닌 matrics로 구해준다
        gradients =tape.gradient(loss, [W[2], W[1], W[0], b[2], b[1], b[0]])#gradient구하기
        #저자리에는 숫자가 들어가야 되기 때문에 w,b이렇게 행렬이 아닌 성분으로 넣어주었다.
        opt.apply_gradients(zip(gradients, [W[2], W[1], W[0], b[2], b[1], b[0]]))

    if epoch % 5==0:
        correct =tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_batch, 1))#y_predict랑 y_batch가 똑같은지 확인
        accuracy =tf.reduce_mean(tf.cast(correct, tf.float32)).numpy()#correct를 실수화 시킨후 평균을 구해서 출력
        print(accuracy)







































