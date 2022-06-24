# -*- coding: utf-8 -*-
"""
Created on Mon May  9 23:43:55 2022

@author: gprua
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
    )

env=gym.make('FrozenLake-v3')#agent가 활동할 수 있는 환경설정

Q=np.zeros([env.observation_space.n,env.action_space.n])

dis=.99  #감마
#경로를 결정하려고 할 때 선택의 문제에 놓인다
#긴 경로의 Q값들은 작아지고 가장짧은 경로의 Q값들은 커진다

num_episodes=100#시행횟수

#각 episode에 대해 스텝과 total reward
rList=[]
for i in range(num_episodes):
    state= env.reset()#agent의 state를 0으로 리셋
    rAll=0
    done=False

    e=1./((i//100)+1)#E-greedy 보완
    #일정확률로 가끔은 최적의 action을 따라가지 않도록 설정

    #한번 수행할때 마다 Q 한칸 업데이트
    while not done:
        #E-greedy를 따라 작은 확률로 랜덤하게 가고, 큰 확률로 높은 Q를 따른는 쪽으로
        if np.random.rand(1)<e:
            action=env.action_space.sample()
            #agent의 움직임 : Random
        else:
            action=np.argmax(Q[state,:])
            #argmax하는 index반환

        print(action)
        #env에 따른 새로운state와 reward
        new_state, reward,done, _ = env.step(action)#움직임에 따른 결과값


        #Q-table update
        Q[state,action]=reward +dis*np.max(Q[new_state,:])

        rAll +=reward
        state= new_state
        # print(Q)


    rList.append(rAll)


print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()