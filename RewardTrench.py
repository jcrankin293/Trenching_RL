import zmq
import numpy as np
import torch as T
import math

# This is the home of the reward function. Apologies for the poor coding.
# Some parts are also redundant. It is only reward that matters here.

use_cuda = T.cuda.is_available()
device = T.device("cuda" if use_cuda else "cpu")
context = zmq.Context()


class MeveaEnvironment(object):
    def __init__(self, no_sensors, actions):
        self.stateSpace = no_sensors

        # context = zmq.Context()

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

        # self.seed()
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def isTerminalState(self, episode, n_episodes):
        if episode == n_episodes:
            return True

    def step(self, action):  # I removed action as it is covered by the ddpg

        # state from cylidner
        state = sensorValues[6:]
        reward = sensorValues[0:8]

        # action = agent.act(state)# I need to check this to send actions. Is it necessary to obtain actions?
        state = state
        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.reward_function(reward)
        return state, reward, done, info

    def reset(self):  # removed (self)
        reset_sim = False
        while not reset_sim:
            reset_sim = False
        else:
            reset_sim = True
        self.goal = self._sample_goal().copy()
        state = self._get_obs()
        return state, reset_sim

    def reward_function(self, dp, sensorValues, d, region_visited, h1, h2, w1, w2, l1, l2,
                        score,prev_mass):  # prev_mass,prev_dump
        # print("Reward  :::   start of func ", reward)
        prev_mass = prev_mass
        mass1 = sensorValues[6]
        DST4 = sensorValues[11]
        DST5 = sensorValues[12]
        DSTT = sensorValues[24]
        slewr = 0
        slew = sensorValues[23]
        episode = 0
        x = round(sensorValues[25], 1)
        y = round(sensorValues[26], 1)
        z = round(sensorValues[27], 1)
        point = str(x) + str(y) + str(z)
        print(point)
        print(len(region_visited))
        trench = len(region_visited)
        dist = 0
        angle_reward = 0
        #
        if y > h2:
            trench = 0

        elif h1 <= y <= h2 and w1 <= x <= w2 and l1 <= z <= l2:  # w 120 h 2 l 60
            check = point in region_visited
            print("In trench list?", check)
            if check is True:
                trench = 0
            else:
                trench = 1
                region_visited.append(point)

        else:
            trench = 0

        if mass1 <= 250:
            if mass1 > prev_mass:
                mass = mass1 - prev_mass
                prev_mass = mass1

            else:
                mass = 0

        else:
            mass = 0

        if d == 0:

            #angle_reward -= abs(sensorValues[20])

            if DST5 > 2.5:
                if DST5<dp:
                    dist = dp-DST5
                    dp = DST5
                else:
                    dist = dp-DST5
            else:
                d = 0
                dp = z
                dist = 0

            #slewr = -abs(slew)

        elif d == -1:

            if z > 2.5:
                if z<dp:
                    dist =dp-z
                    dp = z
                    #angle_reward -= abs(sensorValues[20]+70)
                else:
                    dist = dp-z
            else:
                d = 0
                dp = DSTT
                dist = 0


            #slewr = -abs(slew)

        elif d == 1:

            if DSTT > 2.5:

                if DSTT < dp:
                    dist = dp - DSTT
                    dp = DSTT
                else:
                    dist = dp - DSTT

            else:
                # d = 0
                dp = DST5
                dist = 0
                #angle_reward -= abs(90-sensorValues[20])

            #if -75>= slew >=-180:
             #   slewr = slew + 75

            #if 105>= slew >-75:
             #   slewr = (75+ slew) *-1

            #if 180>= slew > 105:
             #   slewr = (285 - slew) * -1


        if sensorValues[0] <= 0.01:
            time = 0.01
        else:
            time = sensorValues[0]

        score += mass
        # reward = (0.0001*math.exp(-dist/20)) + (mass*0.01)/time + (trench/7812*0.3409) + (sensorValues[7]*0.649/250)/((sensorValues[1]+sensorValues[2]+sensorValues[3])+1)
        reward = (dist) + (((len(region_visited)-1) * 1000)/((max(251,sensorValues[6]))-250)) + mass #+ (slewr/180) #+ angle_reward#/ (max(251, sensorValues[6]) - 250)  # (mass*0.2)/time + + (sensorValues[7])/((sensorValues[1]+sensorValues[2]+sensorValues[3])+1)
        print("========================================")
        print("Dist", dist, "Mass ", mass, "Trench ", trench)
        print("Mass score", score, "D", d, "Slew", slewr)
        print("Prev_mass",prev_mass)
        return dp, reward, d, mass, episode, region_visited, DST5, DSTT, sensorValues[0], score, dist,prev_mass  # score,prev_mass


