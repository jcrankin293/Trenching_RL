#Main script. In here is a client server to communicate with Simulator


import random
import os
import zmq
import numpy as np
import math
import time
import matplotlib.pyplot as plt
# from model import Actor,Critic

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from RewardTrench import MeveaEnvironment
from utils import Agent
from config import *

from numba import jit
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# function optimized to run on gpu

context = zmq.Context()
# Socket to talk to server
print("Connecting to Mevea server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Initialize simulation time
simulationTime = 0
nInputs = 6
indexNo = np.array([float(x) for x in range(6)])

indexNo[0] = 0
indexNo[1] = 0
indexNo[2] = 0
indexNo[3] = 0
indexNo[4] = 0
indexNo[5] = 0

env = MeveaEnvironment(29, 4)#15 with dump target
T.manual_seed(12345)  # was 0
np.random.seed(12345)  # was 0

filename = 'Mevea_TD3_P2P'
figure_file = filename

# Agent alpha = 0.001, beta=0.001, tau=0.005 batch_size = 100 400-300
agent = Agent(alpha=0.001, beta=0.001, input_dims=29, tau=0.005, env=env, batch_size=500, layer1_size=400,
              layer2_size=300,n_actions=4)
#agent.load_models()

status = []

for i_episode in range(n_episodes):
    #    scores_deque = deque(maxlen=100)
    cmax = 0

    max_score = np.Inf
    score = 0
    episode_reward = 0
    episode_steps = 0
    done = False
    d = 0
    dp = 0
    episode = 0
    episodes = 0
    stop_command = 0
    reward = 0
    # Receive the initial state = restarting environment
    print("Retrieving intial state ...")
    # try removing the index command
    indexNo[0] = 0
    indexNo[1] = 0
    indexNo[2] = 0
    indexNo[3] = 0
    indexNo[4] = 0
    indexNo[5] = 0
    inputValues = [f"{x:.17g}" for x in indexNo]
    controlMessage = " ".join(inputValues).encode()

    socket.send(controlMessage)

    sensorMessage: bytes = socket.recv()

    try:
        sensorValues = np.array([float(x) for x in sensorMessage.decode("utf-8").split(" ")])
    except:
        print("Error when converting floats in message")
        print(sensorMessage)
        socket.send(b"error")

    loss = 0
    request = 0
    mass = 0
    prev_mass = 0
    prev_distance = 0
    dump = 0
    region_visited = [str(000)]
    buckmass = 0

    state = [(sensorValues[8] - DST1min) / (DST1max - DST1min), (sensorValues[9] - DST2min) / (DST2max - DST2min),
                        (sensorValues[10] - DST3min) / (DST3max - DST3min), (sensorValues[11] - DST4min) / (DST4max - DST4min),
                        (sensorValues[12] - DST5min) / (DST5max - DST5min), (sensorValues[13] - DST6min) / (DST6max - DST6min),
                        (sensorValues[14] - DST7min) / (DST7max - DST7min), (sensorValues[15] - DST8min) / (DST8max - DST8min),
                        (sensorValues[20] - Bucket_Anglemin) / (Bucket_Anglemax - Bucket_Anglemin),(sensorValues[21] - Dipper_Anglemin) / (Dipper_Anglemax - Dipper_Anglemin),
                        (sensorValues[22] - Boom_Anglemin) / (Boom_Anglemax - Boom_Anglemin),(sensorValues[23] - Slew_Anglemin) / (Slew_Anglemax - Slew_Anglemin),
                        (sensorValues[24] - DSD4min) / (DSD4max - DSD4min),(sensorValues[28] - -105) / (105 - -105),
                        (sensorValues[29] - -50) / (50 - -50),(sensorValues[30] - -20) / (20 - -20),(sensorValues[31] - -50) / (50 - -50),buckmass/1000,sensorValues[7]/80,len(region_visited)/7812,d,dp/15,sensorValues[0]/120.1,prev_distance/15,sensorValues[12]/15,abs(prev_distance-sensorValues[12])/15,(sensorValues[1]/100+sensorValues[2]/100+sensorValues[3]/100),sensorValues[4]/100,prev_mass/251]

    ls = [type(item) for item in state]

    print(ls)
    for request in range(1):  # was 6

        print("Sending request %s …" % request)
        print("d : ", d)
        indexNo[0] = 0
        indexNo[1] = 0
        indexNo[2] = 0
        indexNo[3] = 0
        indexNo[4] = 0
        indexNo[5] = 0
        inputValues = [f"{x:.17g}" for x in indexNo]
        c1 = 1
        c2 = 1
        c3 = 1
        c4 = 1

        # reward = 0
        score = 0  # try this
        x = list()
        y = list()
        z = list()
        w = list()
        new_reward = 0
        h1 = 0.6 #0.2
        h2 = 2.9 #2.5
        l1 = -63.8
        l2 = -60.8
        w1 = -127.5
        w2 = -126.2
        prev_DST5 = 4.7
        prev_DSTT = 0
        prev_time = 0
        mass_score = 0
        mass = 0
        prev_mass = 0
        dump = 0
        prev_dump = 0
        region_visited = [str(000)]
        dp = sensorValues[24]
        distance = 0
        action = [0,0,0,0]

        while episode == 0 or not done:
            action = action
            if total_numsteps<0:#1000 after first
                done = 0
                past_reward = new_reward
                prev_dump = dump
                dump = sensorValues[7]
                prev_distance = distance

                dp, new_reward, d, mass, episodes, region_visiteds, DST5, DSTT, times, mass_score,dist,prev_mass = env.reward_function(
                    dp, sensorValues, d, region_visited, h1, h2, l1, l2, w1, w2, mass_score,prev_mass) #prev_dump, ,prev_mass)
                print("Got R")

                if sensorValues[7]>prev_dump and sensorValues[0]>5:
                    dump = sensorValues[7]
                    new_reward += (dump-prev_dump)*500
                    print("Dump",new_reward)
                    prev_mass = sensorValues[6]

                region_visited = region_visiteds

                prev_DST5 = DST5
                prev_DSTT = DSTT
                prev_time = times

                reward = new_reward  # -past_reward

                if i_episode == n_episodes:
                    stop_command = 1
                    d = 0
                    done = True

                if episodes >= 1:
                    episode = 1
                    d= 0
                    done = True

                if sensorValues[0] >= 120:
                    episode = 1
                    d = 0
                    done = True
                    # reward -= 5000

                if sensorValues[7] >= 250 and sensorValues[0]>10:
                    episode = 1
                    done = True
                    reward += 120 - sensorValues[0]

                if sensorValues[4] > c4:
                    episode = 1
                    reward += -10000
                    d = 0
                    c4 = sensorValues[4]
                    print("bump")
                    done = True

                t = 0

                rewards.append(reward)
                avg_reward = np.mean(rewards[-100:])
                print("Getting action")

                if sensorValues[23]>=-5 and sensorValues[6]<10:
                    if sensorValues[6]<100:
                        if sensorValues[6] ==0:
                            action = [0 + (0.1 * random.uniform(-1, 1)), 1 + (0.1 * random.uniform(-1, 1)),
                                   0.8 + (0.1 * random.uniform(-1, 1)), -0.45*(random.uniform(-1, 1))]
                        else:
                          action = [0+(0.1*random.uniform(-1,1)),-0.5*(0.1*random.uniform(-1,1)),0.7+(0.1*random.uniform(-1,1)),random.uniform(-1,1)]

                elif sensorValues[6]>10 and sensorValues[23]>-70:
                    action = [-0.8+(0.1*random.uniform(-1,1)),-1+(0.1*random.uniform(-1,1)),0+(0.1*random.uniform(-1,1)),1+(0.1*random.uniform(-1,1))]

                elif sensorValues[23]<=-70 and sensorValues[6]>10:
                    action = [0+(0.1*random.uniform(-1,1)),-0.3+(0.1*random.uniform(-1,1)),-0.9+(0.1*random.uniform(-1,1)),-1+(random.uniform(-1,1))]

                if sensorValues[23]<-5 and sensorValues[6]<10 :
                    action = [0.65 + (0.1 * random.uniform(-1, 1)), 0.7+random.uniform(-1, 1),-0.4+
                            (random.uniform(-1, 1)), -0.7+(random.uniform(-1, 1))]

                if action[0]>1:
                    action[0]=1
                elif action[0]<-1:
                    action[0]=-1

                if action[1]>1:
                    action[1]=1
                elif action[1]<-1:
                    action[1]=-1

                if action[2]>1:
                    action[2]=1
                elif action[2]<-1:
                    action[2]=-1

                if action[3]>1:
                    action[3]=1
                elif action[3]<-1:
                    action[3]=-1
                # try removing the index command

            else:

                done = 0
                past_reward = new_reward
                prev_dump = dump
                dump = sensorValues[7]
                prev_distance = distance
                print("Learning")
                agent.learn()
                print("Learnt")

                dp, new_reward, d, mass, episodes, region_visiteds, DST5, DSTT, times, mass_score,dist,prev_mass = env.reward_function(
                    dp, sensorValues, d, region_visited, h1, h2, l1, l2, w1, w2, mass_score,prev_mass) #prev_dump, ,prev_mass)
                print("Got R")

                if sensorValues[7]>prev_dump and sensorValues[0]>5:
                    dump = sensorValues[7]
                    new_reward += (dump-prev_dump)*500
                    prev_mass = sensorValues[6]
                    print("Dump",new_reward)

                region_visited = region_visiteds

                prev_DST5 = DST5
                prev_DSTT = DSTT
                prev_time = times

                reward = new_reward  # -past_reward


                if i_episode == n_episodes:
                    stop_command = 1
                    d = 0
                    done = True

                if episodes >= 1:
                    episode = 1
                    d= 0
                    done = True

                if sensorValues[0] >= 120:
                    episode = 1
                    d = 0
                    done = True
                    # reward -= 5000

                if sensorValues[7] >= 250 and sensorValues[0]>10:
                    episode = 1
                    done = True
                    reward += 120.1 - sensorValues[0]

                if sensorValues[1] >= 0.00001 or sensorValues[2] > 0.00001 or sensorValues[3] > 0.00001:
                    episode = 1
                    d =0
                    reward +=-10000
                    c1 = sensorValues[1]
                    c2 = sensorValues[2]
                    c3 = sensorValues[3]
                    #print("bump")
                    done = True

                if sensorValues[4] > 0.00001:
                    episode = 1
                    reward += -10000
                    d = 0
                    c4 = sensorValues[4]
                    print("bump")
                    #done = True

                t = 0


                rewards.append(reward)
                avg_reward = np.mean(rewards[-100:])

                print("Choosing action")
                #if sensorValues[0]>=0.01 and (sensorValues[0]).is_integer:
                action = agent.choose_action(state)
                #else:
                 #   action = action

            print("Chose action")
            print("Sending action")
            # try removing the index command

            indexNo[0] = action[0]
            indexNo[1] = action[1]
            indexNo[2] = action[2]
            indexNo[3] = action[3]
            indexNo[4] = episode
            indexNo[5] = stop_command
            inputValues = [f"{x:.17g}" for x in indexNo]
            controlMessage = " ".join(inputValues).encode()

            if sensorValues[0] <= 0.1 or (sensorValues[0]).is_integer or episode == 1 or stop_command == 1:
                socket.send(controlMessage)
                print("Sent action")
                sensorMessage: bytes = socket.recv()
                print("recieved sensors")
                try:
                    sensorValues = np.array([float(x) for x in sensorMessage.decode("utf-8").split(" ")])
                except:
                    print("Error when converting floats in message")
                    print(sensorMessage)
                    socket.send(b"error")

            if sensorValues[6] <= 1000:
                buckmass = sensorValues[6]
            else:
                buckmass = 1000

            if sensorValues[6] >= 10 or sensorValues[12]<=2.5:
                d = -1
                distance = sensorValues[11]
            if sensorValues[6] >= 100:
                d = 1
                distance = sensorValues[24]
                #reward +=5000
            if sensorValues[6] <= 10:
                d = 0
                distance = sensorValues[12]

            next_state = [(sensorValues[8] - DST1min) / (DST1max - DST1min), (sensorValues[9] - DST2min) / (DST2max - DST2min),
                        (sensorValues[10] - DST3min) / (DST3max - DST3min), (sensorValues[11] - DST4min) / (DST4max - DST4min),
                        (sensorValues[12] - DST5min) / (DST5max - DST5min), (sensorValues[13] - DST6min) / (DST6max - DST6min),
                        (sensorValues[14] - DST7min) / (DST7max - DST7min), (sensorValues[15] - DST8min) / (DST8max - DST8min),
                        (sensorValues[20] - Bucket_Anglemin) / (Bucket_Anglemax - Bucket_Anglemin),(sensorValues[21] - Dipper_Anglemin) / (Dipper_Anglemax - Dipper_Anglemin),
                        (sensorValues[22] - Boom_Anglemin) / (Boom_Anglemax - Boom_Anglemin),(sensorValues[23] - Slew_Anglemin) / (Slew_Anglemax - Slew_Anglemin),
                        (sensorValues[24] - DSD4min) / (DSD4max - DSD4min),(sensorValues[28] - -105) / (105 - -105),
                        (sensorValues[29] - -50) / (50 - -50),(sensorValues[30] - -20) / (20 - -20),(sensorValues[31] - -50) / (50 - -50),buckmass/1000,sensorValues[7]/80,len(region_visited)/7812,d,dp/15,sensorValues[0]/120.1,prev_distance/15,distance/15,abs(prev_distance-distance)/15,(sensorValues[1]/100+sensorValues[2]/100+sensorValues[3]/100),sensorValues[4]/100,prev_mass/251]

            reward = float(reward)
            episode_steps += 1
            total_numsteps += 1
            print("Remembering")
            if episode_steps > 10:# or (sensorValues[0]).is_integer or episode == 1 or stop_command == 1:
                agent.remember(state, action, reward, next_state, done)
                #status.append([state,reward])
                score += reward

            #elif sensorValues[0]<=0.1:
             #   agent.remember(state, action, reward, next_state, done)
                #status.append([state,reward])
              #  score += reward

            else:
                episode = 0
                d = 0
            print("Remembered")
            #x.append(state,reward)
            x.append(len(region_visited))
            y.append(mass_score)
            z.append((dist/20))
            w.append(reward)

            state = next_state

            if c1 > cmax:
                cmax = c1

            if c2 > cmax:
                cmax = c2

            if c3 > cmax:
                cmax = c3

            if c4 > cmax:
                cmax = c4

            #if episode == 1:
             #   time.sleep(5)
              #  print("Wait a bit")

            print("Episode : ",i_episode)
            print("Total Steps : ",total_numsteps)
            print("Reward", reward)
            print("Score",score)
            #print("dp",dp,"Dump", sensorValues[7],"Trench Length",len(region_visited))
            #print("prev mass",prev_mass,"prev_dump",prev_dump)


        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #plt.plot(x,label='Trench',color = 'r')
        #plt.plot(y,label='Mass', color = 'g')
        #plt.plot(w,label='Reward',color = 'b')
        #plt.plot(z,label='Dist',color = 'm')
        #plt.legend(loc = 'upper left')
        #plt.show()

        if avg_score > best_score and total_numsteps>2000:
            best_score = avg_score
            #agent.save_models()
        if 0.5<=sensorValues[0] and 50<=episode_steps:#<=120:
            reward_total.append([total_numsteps,sensorValues[0], score,d,mass_score,dump,len(region_visited),sensorValues[1],sensorValues[2],sensorValues[3]])
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))

        #if episode == 1:
        print("Pause 20")
        print("Time", time.sleep(20))
        episode = 0
        d = 0
        prev_mass = 0
        prev_dump = 0
        distance = 0
        prev_distance = 0
        dump = 0
        print("Pause Done")
    print("Time", time.sleep(5))


agent.save_models()
print(reward_total)
#print(status)