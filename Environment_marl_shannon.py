from __future__ import division
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

from math import e
from scipy import special

np.random.seed(1234)


class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # center of the grids
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []
        
        self.reward_check=[]
        self.reliability_list=[]
        self.reliability_list_rand=[]
        self.reward_check_rand=[]
        self.network_avg_throughput=[]
        self.network_avg_throughput_rand=[]
        self.metric=0
        self.metric_rand=0
        
        self.demand = []
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []

        self.V2I_power_dB = 23  # dBm
        self.V2V_power_dB_List = [23 ,20,15,10]  # the power levels
        self.V2I_power = 10 ** (self.V2I_power_dB)
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.n_RB = n_veh
        self.n_Veh = n_veh
        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(180e3)  ## int(1e6)  # bandwidth per RB, 1 MHz
   
        

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd' # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============
        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(self.n_neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j + 1])
            destination = self.vehicles[i].neighbors

            self.vehicles[i].destinations = destination

    def renew_channel(self):
        """ Renew slow fading channel """

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape))/ math.sqrt(2))

  
    
    def Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                # if not self.active_links[i, j]: ## will skip when  self.active_links[i, j] is 0 or false, intially active links=[ 1 1 1 1]
                #     continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        
        # print("action values before", actions, "self.active_links", self.active_links)
        # actions[(np.logical_not(self.active_links))] = -1 # inactive links will not transmit regardless of selected power levels
        #                                                     # replaces the RB action ( with -1) for which corresponding link is inactive (False)
        
        # print("action values after", actions)
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs e-g, [1, 0,0,1] meansfor 0RB and 3RB link active [[1 0] [2 0]] and [[0 0][3 0]]
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))  ## units are (bits/sec)/Hz
        # print( "V2V rate is", V2V_Rate)
        #############################################################################
        temp=1/ np.power ((1+ np.divide(V2V_Signal, self.V2V_Interference)),2)
        dispersion= 1-temp
        blocklength=self.time_fast*self.bandwidth  #= int(1e6)  # bandwidth per RB, 1 MHz
    
       
        qos= np.array([4e-5 ,3e-5 , 2e-5, 1e-5]).reshape(V2V_Rate.shape)
        # qos= np.array([8e-4, 7e-4, 6e-3, 5e-4, 4e-3 ,3e-4 , 2e-3, 1e-4]).reshape(V2V_Rate.shape)
        
        effec_rate=  V2V_Rate - (np.sqrt(dispersion/blocklength))* (np.sqrt(2)*special.erfinv(1-2*( qos)))/np.log(2)  # Q^(-1)(f) = np.sqrt(2)*special.erfinv(1-2*(error))
        
        # print( "effec_rate", effec_rate)
       
        C_minus_R=  V2V_Rate- 100/blocklength
        fun= math.log(2)*(np.sqrt(blocklength/dispersion))* C_minus_R                 
        error_all= 0.5 - 0.5*special.erf(fun/np.sqrt(2)) # Q(f) = 0.5 - 0.5 erf(f/sqrt(2))                
        
        # print ("V2V rates  are", V2V_Rate.reshape(-1))
        # print ("error probabilities are", error_all.reshape(-1))
        # print ("SRNs are ",(np.divide(V2V_Signal, self.V2V_Interference)).reshape(-1) , " dispersion are",  dispersion.reshape(-1))
        
      
        reward_elements = V2V_Rate/10 
       
        reliability= np.where(np.array( error_all) < qos, 1, 0) ## reward of 1/0 if element in error_all </> qos
        
        # print("reliability is",reliability.reshape(-1))
        # if np.max(error_all)<= 1e-5:
        #     reward_elements +=1
        reward_elements[ reliability.reshape(effec_rate.shape) > 0] = 1  ### a reward element of 1 if reliable transmissssion
        
        # if np.max(error_all) <np.min(qos):
        #    reward_elements+=0.01 
        
        # print("reward elements for summation is",reward_elements.reshape(-1))
        self.reliability_list.append(np.max(error_all))
        self.reward_check.append(reward_elements)
        
        self.network_avg_throughput.append( np.sum((100/blocklength)*(1- error_all) )) ## sum(R*(1-epsilon))
        
        if np.max(error_all) < 1e-3:
            self.metric=1
        else:
            self.metric=0
            
        
        # self.metric = np.where(np.array(np.max(error_all)) < 1e-3,1,0)  # metrics for XAI
        #############################################################################
        # self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        # self.demand[self.demand < 0] = 0 # eliminate negative demands

        # self.individual_time_limit -= self.time_fast
        # reward_elements = V2V_Rate/10
        # reward_elements[self.demand <= 0] = 1 
        
        
        # print("demand is ",self.demand)
        # self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 # transmission finished, turned to "inactive"
        
        
        # reward_elements= effec_rate
        
        return V2I_Rate, V2V_Rate,  effec_rate,  reward_elements
    

    
    
    def Compute_Performance_Reward_Test_rand(self, actions_power):
        """ for random baseline computation """

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_random = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference_random))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_random = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))  ## units are (bits/sec)/Hz
        # print( "V2V rate is", V2V_Rate)
        #############################################################################
        temp=1/ np.power ((1+ np.divide(V2V_Signal, self.V2V_Interference)),2)
        dispersion= 1-temp
        blocklength=self.time_fast*self.bandwidth  #= int(1e6)  # bandwidth per RB, 1 MHz
    
       
        qos= np.array([4e-5 ,3e-5 , 2e-5, 1e-5]).reshape(V2V_Rate.shape)
        effec_rate=  V2V_Rate - (np.sqrt(dispersion/blocklength))* (np.sqrt(2)*special.erfinv(1-2*( qos)))/np.log(2)  # Q^(-1)(f) = np.sqrt(2)*special.erfinv(1-2*(error))
        
        # print( "effec_rate", effec_rate)
       
        C_minus_R=  V2V_Rate- 100/blocklength
        fun= math.log(2)*(np.sqrt(blocklength/dispersion))* C_minus_R                 
        error_all= 0.5 - 0.5*special.erf(fun/np.sqrt(2)) # Q(f) = 0.5 - 0.5 erf(f/sqrt(2))                
        

        reward_elements = V2V_Rate/10
        reliability= np.where(np.array( error_all) < qos, 1, 0) ##  reward of 1/0 if element in error_all </> qos

        reward_elements[ reliability.reshape(effec_rate.shape) > 0.1] = 1  ### a reward element of 1 if payload is zero/negative means all bits delivered equivalent to beta
        
        # print("reward elements for summation is",reward_elements.reshape(-1))
        self.reliability_list_rand.append(np.max(error_all))
        self.reward_check_rand.append(reward_elements)
        
        self.network_avg_throughput.append( np.sum((100/blocklength)*(1- error_all) )) ## sum(R*(1-epsilon))
        
        # print('errors are', (error_all).reshape(1,-1))
        
        if np.max(error_all) < 1e-3:
            
            # print("yes it is true")
            # print('errors are', np.max(error_all))
            self.metric_rand=1
        else:
            # print("yNot true")
            # print('errors are', np.max(error_all))
  
            self.metric_rand=0
        
        # self.metric_rand = ((np.max(error_all) < 1e-3) ) # metrics for XAI
            

        return V2I_Rate, V2V_Rate,  effec_rate, reward_elements
    
    
    
    

    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        # channel_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]]
                                                                                   - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)


    
    def act_for_training(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, effec_rate,reward_elements= self.Compute_Performance_Reward_Train(action_temp)
        # print("V2I rate is: ", V2I_Rate.reshape(-1))
        # print("V2V rate is: ",V2V_Rate.reshape(-1))
        # print("effective rate is :",effec_rate.reshape(-1))
        lambdda = 0.1
        reward = lambdda * np.sum(V2I_Rate) / (self.n_Veh * 10) + (1 - lambdda) * np.sum(reward_elements) / (self.n_Veh * self.n_neighbor)

        return reward  
    

    # def act_for_testing(self, actions):

    #     action_temp = actions.copy()
    #     V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
    #     V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates

    #     return V2I_Rate, V2V_success, V2V_Rate
    
    
    def act_for_testing(self, actions):

        # action_temp = actions.copy()
        # V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        # V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates
        
        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, effec_rate,reward_elements= self.Compute_Performance_Reward_Train(action_temp)
        # print("V2I rate is: ", V2I_Rate.reshape(-1))
        # print("V2V rate is: ",V2V_Rate.reshape(-1))
        # print("effective rate is :",effec_rate.reshape(-1))
     
        V2V_success = np.sum(reward_elements) / (self.n_Veh * self.n_neighbor) ## 100% (4/4)  means all vehicles satisfy the QOS 75%(3/4) means 3/4 satisfy

        return V2I_Rate, V2V_success, V2V_Rate,  effec_rate
    


    # def act_for_testing_rand(self, actions):

    #     action_temp = actions.copy()
    #     V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
    #     V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  # V2V success rates

    #     return V2I_Rate, V2V_success, V2V_Rate
    
    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, effec_rate, reward_elements = self.Compute_Performance_Reward_Test_rand(action_temp)
        V2V_success = np.sum(reward_elements) / (self.n_Veh * self.n_neighbor) ## 100% (4/4)  means all vehicles satisfy the QOS 75%(3/4) means 3/4 satisfy

        return V2I_Rate, V2V_success, V2V_Rate,  effec_rate

    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

        # random baseline
        self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
        
    def plot_training_rewards(self, rewards):
            # Calculate a moving average of rewards
            window_size = 500
            moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards) - window_size + 1)]
        
            # Create a time axis for x-axis
            episodes = range(len(rewards))
        
            # Plot the rewards and moving average
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, self.smooth(rewards,30), label='Training Rewards', alpha=0.4)
            plt.plot(episodes[window_size - 1:], moving_avg, label=f'Moving Average (window={window_size})', color='r')
            plt.xlabel('Training step')
            plt.ylabel('Reward per time step ')
            plt.legend()
            # plt.title('Training Rewards in DRL')
            plt.grid(True)
            plt.show()
            

    
    def smooth(self, Episode_Return,WSZ):
        # a: NumPy 1-D array containing the data to be smoothed
        # WSZ: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        out0 = np.convolve(Episode_Return,np.ones(WSZ,dtype=int),'valid')/WSZ    
        r = np.arange(1,WSZ-1,2)
        start = np.cumsum(Episode_Return[:WSZ-1])[::2]/r
        stop = (np.cumsum(Episode_Return[:-WSZ:-1])[::2]/r)[::-1]
        return np.concatenate((  start , out0, stop  ))
    
    
    def create_state_space_dictionary(self, names, elements):
        if len(names) != len(elements):
            raise ValueError("Names and elements must have the same length.")
        
        state_space = {}
        for i in range(len(names)):
            state_space[names[i]] = elements[i]
    
        return state_space

       
    
    def get_first_n_items(self,dictionary, n):
        return dict(list(dictionary.items())[:n])




