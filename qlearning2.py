# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:26:25 2022

@author: gprua
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
import gym

#q-learning을 위한 env설정
env = gym.make('CartPole-v1')
goal_steps = 500

while True:
  obs = env.reset()
  for i in range(goal_steps):
    obs, reward, done, info = env.step(random.randrange(0, 2))
    if done: break
    env.render()

#학습데이터 생성
def data_preparation(N, K, f, render=False):
  game_data = []
  for i in range(N):
    score = 0
    game_steps = []
    obs = env.reset()
    for step in range(goal_steps):
      if render: env.render()
      action = f(obs)
      game_steps.append((obs, action))
      obs, reward, done, info = env.step(action)
      score += reward
      if done:
        break
    game_data.append((score, game_steps))
  
  game_data.sort(key=lambda s:-s[0])
  
  training_set = []
  for i in range(K):
    for step in game_data[i][1]:
      if step[1] == 0:
        training_set.append((step[0], [1, 0]))
      else:
        training_set.append((step[0], [0, 1]))

  print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
  if render:
    for i in game_data:
      print("Score: {0}".format(i[0]))
  
  return training_set
training_data=data_preparation(1000,50, lambda s:random.randrange(0,2))

#데이터 학습
def build_model():
    model= Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='mse',optimizer=Adam())
    return model

def train_model(model, training_set):
    X=np.array([i[0] for i in training_set]).reshape(-1,4)
    y=np.array([i[1] for i in training_set]).reshape(-1,2)
    model.fit(X,y,epochhs=10)#model 학습

#상위의 데이터를 가져오기
if __name__ == '__main__':
    N=1000
    K=50
    self_play_count=10
    model = build_model()
    training_data=data_preparation(N,K,lambda s:random.randrange(0,2))
    train_model(model,training_data)

def predictor(s):
     return np.random.choice([0,1], p=model.predict(s.reshape(-1,4))[0])

for i in range(self_play_count):
        K=(N//9+K)//2
        training_data =data_preparation(N,K,predictor)
        train_model(model,training_data)

data_preparation(100,100,predictor,True)