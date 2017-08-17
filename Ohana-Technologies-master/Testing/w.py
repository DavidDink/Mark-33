# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
#
"""
            optionAction = [[2, 2],[2, 1.5], [2, 1], [2, 0.5], [2, 0], [2, -0.5], [2,-1], [2,-1.5], [2,-2],
            [1.5,2],[1.5,1.5], [1.5,1], [1.5,0.5], [1.5,0], [1.5,-0.5], [1.5,-1], [1.5,-1.5], [1.5,-2],
            [1, 2], [1, 1.5], [1, 1], [1, 0.5], [1, 0], [1, -0.5], [1, -1], [1, -1.5], [1, -2],
            [0.5, 2], [0.5, 1.5], [0.5, 1], [0.5, 0.5], [0.5, 0], [0.5, -0.5], [0.5, -1], [0.5, -1.5], [0.5, -2],
            [0, 2], [0, 1.5], [0, 1], [0, 0.5], [0, 0], [0, -0.5], [0, -1], [0, -1.5], [0, -2],
            [-2, 2], [-2, 1.5], [-2, 1], [-2, 0.5], [-2, 0], [-2, -0.5], [-2, -1], [-2, -1.5],[-2, -2],
            [-1.5, 2], [-1.5, 1.5], [-1.5, 1], [-1.5, 0.5], [-1.5, 0], [-1.5, -0.5], [-1.5, -1], [-1.5, -1.5], [-1.5, -2],
            [-1, 2], [-1, 1.5], [-1, 1], [-1, 0.5], [-1, 0], [-1, -0.5], [-1, -1], [-1, -1.5], [-1, -2],
            [-0.5, 2], [-0.5, 1.5], [-0.5, 1], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [-0.5, -1], [-0.5, -1.5], [-0.5, -2]]

"""


# --- enable this to run on GPU
# import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import random, numpy, math

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from py4j.java_gateway import JavaGateway
import numpy as np
import h5py
#import matplotlib.pyplot as plt



class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt#discrete in Open AI GYM

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(24, input_dim=16, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(9, activation='linear'))

        """
        model.add(Convolution1D(activation="relu", input_dim=16, units=64))
        model.add(Convolution1D(subsample=(, ), activation='relu'))
        model.add(Convolution1D(subsample=(, ), activation='relu' ))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(activation="linear", units=9))
        """
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s): # changed from predictOne to PredictTwo
        return self.predict(s.reshape(1, self.stateCnt)).flatten()


# -------------------- MEMORY Checked.--------------------------
class Memory:  # stored as ( state, action, reward, new_state )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n) #returns n random samples form the memory.



# -------------------- AGENT ---------------------------
#Neural Network for Parameters
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001   #how fast the agent becomes less random and chooses the best q-value.


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt) #Initiates the Brain
        self.memory = Memory(MEMORY_CAPACITY) # Initiates the Memory

    def act(self, s):

        # Playing around (guessing)
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        # If it is sure enough with itself, it makes a real prediction
        else:
            print("not guessing. I know what I am doing.")
            s = np.array(s)
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample) #stores the experience into the internal array, making sure that it does not exceed its capacity.

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        # Get a list of previous intervals
        batch = self.memory.sample(BATCH_SIZE)
        # Length of batch
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            sample = batch[i]
            current_state_sample = sample[0];
            current_output = sample[1];
            current_reward = sample[2];
            prev_state_sample = sample[3]

            t = p[i]
            if current_reward < 0:
                t[int(current_output)] = current_reward
            else:
                t[int(current_output)] = current_reward + GAMMA * numpy.amax(p_[i])


            x[i] = current_state_sample
            y[i] = t

        self.brain.train(x, y)

# Classify net output
""""
optionAction = [[0, 0], [2, 2],[2, 1.5], [2, 1], [2, 0.5], [2, 0], [2, -0.5], [2,-1], [2,-1.5], [2,-2],
            [1.5,2],[1.5,1.5], [1.5,1], [1.5,0.5], [1.5,0], [1.5,-0.5], [1.5,-1], [1.5,-1.5], [1.5,-2],
            [1, 2], [1, 1.5], [1, 1], [1, 0.5], [1, 0], [1, -0.5], [1, -1], [1, -1.5], [1, -2],
            [0.5, 2], [0.5, 1.5], [0.5, 1], [0.5, 0.5], [0.5, 0], [0.5, -0.5], [0.5, -1], [0.5, -1.5], [0.5, -2],
            [0, 2], [0, 1.5], [0, 1], [0, 0.5], [0, -0.5], [0, -1], [0, -1.5], [0, -2],
            [-2, 2], [-2, 1.5], [-2, 1], [-2, 0.5], [-2, 0], [-2, -0.5], [-2, -1], [-2, -1.5],[-2, -2],
            [-1.5, 2], [-1.5, 1.5], [-1.5, 1], [-1.5, 0.5], [-1.5, 0], [-1.5, -0.5], [-1.5, -1], [-1.5, -1.5], [-1.5, -2],
            [-1, 2], [-1, 1.5], [-1, 1], [-1, 0.5], [-1, 0], [-1, -0.5], [-1, -1], [-1, -1.5], [-1, -2],
            [-0.5, 2], [-0.5, 1.5], [-0.5, 1], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [-0.5, -1], [-0.5, -1.5], [-0.5, -2]]
"""

optionAction = [[0,0],[2, 2],[-2,-2], [2,0], [2,-2], [0, 2], [0,-2], [-2, 0], [-2,2]]
def classify_output(output):
    if output < 0 or output >= 9:
        print('Error, output is out of range: ' + str(output))
        exit(1)
    print ('Output = ' + str(output))
    new_output = optionAction[int(output)]

    new_output[0] = float(new_output[0])
    new_output[1] = float(new_output[1])
    return new_output

def within_range(val, desired_val, allowed_deviation):
    return abs(desired_val - val) <= allowed_deviation

def get_dist_to_range(val, min_val, max_val):
    return min(abs(min_val - val), abs(max_val - val))

def eval_temp_humidity_reward(prev_val, curr_val, desired_val):
    allowed_deviation = 4  # Hard coded - dangerous
    val_in_comfort_zone = within_range(curr_val, desired_val, allowed_deviation)
    if val_in_comfort_zone:
        return 1 # Default reward for when val is in comfort zone
    else:
        return -1
    """
    min_comfortable_val = desired_val - allowed_deviation
    max_comfortable_val = desired_val + allowed_deviation
    # Always a neg val
    dist_to_desired_val = - get_dist_to_range(curr_val, min_comfortable_val, max_comfortable_val)
    prev_dist_to_desired_val = - get_dist_to_range(prev_val, min_comfortable_val, max_comfortable_val)

    
    # Calculate change in valerature

    
    d_val = abs(curr_val - prev_val) * 4
    if dist_to_desired_val < prev_dist_to_desired_val:
        d_val *= -1

    total_val_reward = dist_to_desired_val + d_val
    return total_val_reward
    """

def eval_reward(prev_state, curr_state, curr_net_cost, best_net_cost):
    # Evaluate temp reward
    desired_temp = curr_state[1]
    prev_temp = prev_state[2]
    curr_temp = curr_state[2]
    temp_reward = eval_temp_humidity_reward(prev_temp, curr_temp, desired_temp)

    # Evaluate humidity reward
    desired_hum = curr_state[5]
    prev_hum = prev_state[6]
    curr_hum = curr_state[6]
    hum_reward = eval_temp_humidity_reward(prev_hum, curr_hum, desired_hum)

    # Net Cost
    money_cost_reward = 0

    if curr_net_cost <= best_net_cost:
        money_cost_reward = 1
    else:
        money_cost_reward = -1

    # Evaluate final reward
    total_reward = temp_reward + hum_reward + money_cost_reward
    return total_reward


# -------------------- MAIN ----------------------------
gateway = JavaGateway()
sim = gateway.entry_point
sim.setFileSavePath('/home/tonythetiger/Desktop/EVERYDAY/')


stateCnt = 16
actionCnt = 9

agent = Agent(stateCnt, actionCnt)
num_episode = 1000
num_ep = 24
try:
    sim.setStandardSeed(1345)
    sim.resetSession()
    prev_state = sim.getEnvironmentState()

    best_net_cost = 100000
    curr_net_cost = 0
    curr_cost_reward = 0

    net_reward = 0
    rewards = []

    print ('Just started running the net! Yay!')

    for i in range(num_episode):
        sim.resetSession()
        for j in range(num_ep):
            curr_state = list(sim.getEnvironmentState())

            raw_output = float(agent.act(curr_state))
            print('raw output:', str(raw_output))
            classified_output = classify_output(raw_output)

            # Create method for Step
            sim.inputAction(classified_output[0], classified_output[1])

            money_cost = curr_state[-4]
            comfort_penalty = curr_state[-5]
            net_penalty = curr_state[-1]

            # Reward system
            reward = eval_reward(prev_state, curr_state, curr_net_cost, best_net_cost)


            agent.observe([curr_state, raw_output, reward, prev_state])
            # Reflection - analize past acitons
            agent.replay()

            prev_state = curr_state
            net_reward += reward
            rewards.append(reward)

            # Update current net cost
            curr_net_cost = curr_state[-2]

            print ('episode: ' + str(i) + ' ep: ' + str(j))
            print ('current state' + str(curr_state))
            print ('Net cost: ' + str(curr_net_cost))


            print("Reward Received:", reward)

        # Update net cost

        #save data file
        sim.saveDataFile()
        best_net_cost = min(best_net_cost, curr_net_cost)
finally:
    agent.brain.model.save("cartpole-basic.h5")

print('Process completed')

#path = gateway.wakeupLester(num_episode)
#create_Graph(path)
gateway.shutdown()
"""
plt.plot(range(len(rewards)), rewards)
plt.plot(True)
plt.xlabel('Time')
plt.ylabel('Reward')
plt.savefig('/Users/ML_Work/Desktop/CSV DATA EVERDAY/graph.png')
plt.show()
"""