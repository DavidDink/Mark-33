#-------------------- BRAIN ---------------------------
import random, numpy, math, gym

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from Agent import Agent
from Brain import Brain
from Environment import Environment
from Memory import Memory


#-------------------- PARAMETERS ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay


PROBLEM = 'CartPole-v0'
env = Environment()

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")

