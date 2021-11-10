import zmq
import numpy as np
import torch as T
import math

#This is the home of the reward function. Apologies for the poor coding.
# Some parts are also redundant. It is only reward that matters here.

use_cuda = T.cuda.is_available()
device   = T.device("cuda" if use_cuda else "cpu")
context = zmq.Context()


class MeveaEnvironment(object):
    def __init__(self,no_sensors,actions):
        self.stateSpace = no_sensors

        #context = zmq.Context()

        #  Socket to talk to server
        print("Connecting to Mevea server…")
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")

        simulationTime = 0
        self.action_size = np.array([float(x) for x in range(actions)])

        self.min_action = -1.0
        self.max_action = 1.0

        self.action_space = actions
        self.observation_space = 18

        #self.seed()
        #self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def isTerminalState(self,episode,n_episodes):
        if episode == n_episodes:
            return True

    def step(self,action): # I removed action as it is covered by the ddpg

        #state from cylidner
        state = sensorValues[6:]
        reward = sensorValues[0:8]
        
        #action = agent.act(state)# I need to check this to send actions. Is it necessary to obtain actions? 
        state = state
        done = False
        
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.reward_function(reward)
        return state, reward, done, info
        
    def reset(self): # removed (self)
        reset_sim = False
        while not reset_sim:
            reset_sim = False
        else:
            reset_sim = True
        self.goal = self._sample_goal().copy()
        state = self._get_obs()
        return state,reset_sim


    def reward_function(self, dp,sensorValues, d, region_visited, h1, h2, w1, w2, l1, l2,score):#prev_mass,prev_dump
        # print("Reward  :::   start of func ", reward)
        #prev_mass = prev_mass
        mass1 = sensorValues[6]
        DST5 = sensorValues[12]
        DSTT = sensorValues[24]
        episode = 0
        x = round(sensorValues[25], 1)
        y = round(sensorValues[26], 1)
        z = round(sensorValues[27], 1)
        point = str(x)+str(y)+str(z)
        print(point)
        print(len(region_visited))
        trench = 0
        dist = 0
        #
        if y > h2:
            trench = 0

        elif h1 <= y <= h2 and w1 <= x <= w2 and l1 <= z <= l2:#w 120 h 2 l 60
            check = point in region_visited
            print("In trench list?",check)
            if check is True:
                trench = 0
            else:
                trench += 1
                region_visited.append(point)

        else:
            trench = 0

        if mass1<=250:
            mass = sensorValues[6]/250

        else:
            mass = 1

        if d == 0:

           if DST5>2.5:
            #   if DST5<dp:
             #      dist+=dp-DST5
              #     dp = DST5
               #else:
                #   dist+= dp-DST5
            dist += DST5+2.5
           else:
               d = 0
               dp = DSTT
               dist = 0

        else:

           if DSTT>2.5:

               #if DSTT < dp:
                #   dist += dp - DSTT
                 #  dp = DSTT
               #else:
                #   dist += dp - DSTT
               dist += DSTT+2.5

           else:
              # d = 0
               dp = DST5
               dist = 0

#Distance Test
        #
        # if d == 0:
        #
        #     if DSTT>2.5:
        #         if DSTT<dp:
        #             dist+=dp-DSTT
        #             dp = DSTT
        #         else:
        #             dist+= dp-DSTT
        #         #dist += -1+math.exp(-(DST2-2.5)/15)
        #
        #     else:
        #         d = 0
        #         dist = 0
        #
        # if d == 1:
        #
        #     if DST2>2.5:
        #
        #         if DST2 < dp:
        #             dist += dp - DST2
        #             dp = DST2
        #         else:
        #             dist += dp - DST2
        #         #dist += -1+math.exp(-(DST2-2.5)/15)
        #
        #     elif DST2 <= 2.5 and sensorValues[0] > 2:
        #         dist = 10000
        #         episode = 1
        #
        #     else:
        #         d = 0
        #         dist = 0
        #

        #if sensorValues[6] == 0:
         #   time = 1
        #else:
         #   time += 0.0036
        if sensorValues[0]<=0.01:
            time = 0.01
        else:
            time = sensorValues[0]

 

        score += mass
        #reward = (0.0001*math.exp(-dist/20)) + (mass*0.01)/time + (trench/7812*0.3409) + (sensorValues[7]*0.649/250)/((sensorValues[1]+sensorValues[2]+sensorValues[3])+1)
        reward = (-1+math.exp(-dist/18))+(trench*100)/(max(251,sensorValues[6])-250)# (mass*0.2)/time + + (sensorValues[7])/((sensorValues[1]+sensorValues[2]+sensorValues[3])+1)
        print("========================================")
        print("Dist",dist/20,"Mass ", mass,"Trench ", trench)
        print("Mass score", score, "D",d)
        return dp,reward, d, mass, episode, region_visited,DST5,DSTT,sensorValues[0],score,dist#score,prev_mass


