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

import random, numpy, math, gym

# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from py4j.java_gateway import JavaGateway


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt#discrete in Open AI GYM

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(activation="relu", input_dim=16, units=64))
        model.add(Dense(activation="linear", units=81))

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
LAMBDA = 0.001  #how fast the agent becomes less random and chooses the best q-value.


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
optionAction = [[2, 2],[2, 1.5], [2, 1], [2, 0.5], [2, 0], [2, -0.5], [2,-1], [2,-1.5], [2,-2],
            [1.5,2],[1.5,1.5], [1.5,1], [1.5,0.5], [1.5,0], [1.5,-0.5], [1.5,-1], [1.5,-1.5], [1.5,-2],
            [1, 2], [1, 1.5], [1, 1], [1, 0.5], [1, 0], [1, -0.5], [1, -1], [1, -1.5], [1, -2],
            [0.5, 2], [0.5, 1.5], [0.5, 1], [0.5, 0.5], [0.5, 0], [0.5, -0.5], [0.5, -1], [0.5, -1.5], [0.5, -2],
            [0, 2], [0, 1.5], [0, 1], [0, 0.5], [0, 0], [0, -0.5], [0, -1], [0, -1.5], [0, -2],
            [-2, 2], [-2, 1.5], [-2, 1], [-2, 0.5], [-2, 0], [-2, -0.5], [-2, -1], [-2, -1.5],[-2, -2],
            [-1.5, 2], [-1.5, 1.5], [-1.5, 1], [-1.5, 0.5], [-1.5, 0], [-1.5, -0.5], [-1.5, -1], [-1.5, -1.5], [-1.5, -2],
            [-1, 2], [-1, 1.5], [-1, 1], [-1, 0.5], [-1, 0], [-1, -0.5], [-1, -1], [-1, -1.5], [-1, -2],
            [-0.5, 2], [-0.5, 1.5], [-0.5, 1], [-0.5, 0.5], [-0.5, 0], [-0.5, -0.5], [-0.5, -1], [-0.5, -1.5], [-0.5, -2]]

def classify_output(output):
    if output < 0 or output > 80:
        print('Error, output is out of range: ' + str(output))
        exit(1)
    print ('Output = ' + str(output))
    new_output = optionAction[int(output)]
    print(type(output))
    new_output[0] = float(new_output[0])
    new_output[1] = float(new_output[1])
    return new_output
# -------------------- MAIN ----------------------------
gateway = JavaGateway()
sim = gateway.entry_point


stateCnt = 16
actionCnt = 81

agent = Agent(stateCnt, actionCnt)
num_episode = 1000
try:
    sim.setStandardSeed(1345)
    sim.resetSession()
    sim.inputAction(0.0, 0.0)

    prev_state = sim.getEnvironmentState()
    net_reward = 0

    print ('Just started running the net! Yay!')
    for j in range(num_episode):
        for i in range(1000000000):
            curr_state = list(sim.getEnvironmentState())

            raw_output = float(agent.act(curr_state))
            classified_output = classify_output(raw_output)

            # Create method for Step
            sim.inputAction(classified_output[0], classified_output[1])

            money_cost = curr_state[-4]
            comfort_penalty = curr_state[-5]
            net_penalty = curr_state[-1]

            # Reward system
            if money_cost == 0 and comfort_penalty == 0:
                reward = 1
            else:
                reward = net_penalty * -1

            agent.observe([curr_state, raw_output, reward, prev_state])
            # Reflection - analize past acitons
            agent.replay()

            prev_state = curr_state
            net_reward += reward

            print("Reward Received:", str(i), reward)
            print("Total reward:", str(i), net_reward)
finally:
    agent.brain.model.save("cartpole-basic.h5")

print('Process completed')
gateway.shutdown()
