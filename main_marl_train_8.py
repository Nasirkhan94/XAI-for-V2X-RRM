from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
from math import log, e
import time 
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import Environment_marl_8

from replay_memory import ReplayMemory
# import sys
import matplotlib.pyplot as plt
my_config =tf.compat.v1.ConfigProto() 
my_config.gpu_options.allow_growth=True   


import pandas  as pd
import shap
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 0.99
        self.double_q = False
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'marl_model_Original_8'


n_veh = 8
n_neighbor = 1
n_RB = n_veh

env = Environment_marl_8.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 2000## originally 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.01
epsi_anneal_length = int(0.5*n_episode)
mini_batch_step = n_step_per_episode*5
target_update_step = n_step_per_episode*8 

n_episode_test = 100# test episodes  



######################################################

def get_first_n_items(dictionary, n):
        return dict(list(dictionary.items())[:n])

def get_state(env, idx=(0,0) ):
    """ Get state from the environment """
        
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :])


    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :])
    
    
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :])
    
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference))
##########################################################################
# -----------------------------------------------------------
## Simplified MADRL
# n_hidden_1 = len(get_state(env))*5 +8
# n_hidden_2 = len(get_state(env))*3
# n_hidden_3 = len(get_state(env))*2
# n_input = len(get_state(env=env))
# n_output = n_RB * len(env.V2V_power_dB_List)

n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.V2V_power_dB_List)

g = tf.Graph()
with g.as_default():
    # ============== Training network ========================
    x = tf.placeholder(tf.float32, [None, n_input])

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y=  tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)

    # ==================== Prediction network ========================
    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=50) ## defines the maximum number of latest check points to keep.

   
    def count_number_trainable_params():
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in tf.trainable_variables():
            shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
            current_nb_params = get_nb_params_shape(shape)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params*int(dim)
        return nb_params 
    
    total_parameters_per_agent= count_number_trainable_params()
    print("Number of trainable parameters: %d" % total_parameters_per_agent)


def predict(sess, s_t, ep, test_ep = False):

    n_power_levels = len(env.V2V_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def q_learning_mini_batch(current_agent, current_sess):
    """ Training a sampled mini-batch """

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  # double q-learning
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_q_network(sess):
    """ Update target q network once in a while """

    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


def save_models(sess, model_path):
    """ Save models to the current directory with the name filename """
    current_dir = os.getcwd()
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)



def load_models(sess, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):
    """ debug """

    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))



# --------------------------------------------------------------
agents = []
sesses = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    # print("Number of trainable parameters: %d" % count_number_trainable_params())
    sess.run(init)
   
    sesses.append(sess)

# ------------------------- Training -----------------------------

strategy_time_all_agents_per_step =  []
time_optimization_at_each_slot_all_agents=  []

if IS_TRAIN:
    record_reward = np.zeros([n_episode*n_step_per_episode, 1])
    return_log=[]
    record_loss = []
    state_log_XAI=[]

    for i_episode in range(n_episode):

        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final/100
        if i_episode%100 == 0: ## 100 episodess
            env.renew_positions() # update vehicle position
            env.renew_neighbor()
            env.renew_channel() # update channel slow fading--->> updated every 100 episodes and not on each step
            env.renew_channels_fastfading() # update channel fast fading


        for i_step in range(n_step_per_episode):
            time_step = i_episode*n_step_per_episode + i_step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32') ## matrix of 4x1x2
            
            time_calculating_strategy_takes = []
            for i in range(n_veh):
                for j in range(n_neighbor):
                    # state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    state = get_state(env, [i, j] )
                    state_old_all.append(state)
                    start_time = time.time()
                    action = predict(sesses[i*n_neighbor+j], state, epsi)
                    time_calculating_strategy_takes.append(time.time()-start_time )
                    action_all.append(action)

                    action_all_training[i, j, 0] = action % n_RB  # chosen RB
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB)) # power level
         
                   
            strategy_time_all_agents_per_step.append(np.sum(time_calculating_strategy_takes))
            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            record_reward[time_step] = train_reward
            state_log_XAI.append(state)
         
            env.renew_channels_fastfading()  ## fast fading channels renewed on each time step (1ms) in each episode
            env.Compute_Interference(action_temp)
            

            time_optimization_at_each_slot_takes = []
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    # state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    state_new = get_state(env, [i, j])
                    agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)  # add entry to this agent's memory                                                      
                    # training this agent
                    if time_step % mini_batch_step == mini_batch_step-1:
                        start_time=time.time()
                        loss_val_batch = q_learning_mini_batch(agents[i*n_neighbor+j], sesses[i*n_neighbor+j])
                        time_optimization_at_each_slot_takes.append(time.time()-start_time)
                        record_loss.append(loss_val_batch)
                        if i == 0 and j == 0:
                            print('step:', time_step, 'agent',i*n_neighbor+j, 'loss', loss_val_batch)
                    if time_step % target_update_step == target_update_step-1:
                        update_target_q_network(sesses[i*n_neighbor+j])
                        if i == 0 and j == 0:
                            print('Update target Q network...')
                            # V2I_Rate, V2V_Rate, effec_rate= env.Compute_Performance_Reward_Train(action_temp)
              
            time_optimization_at_each_slot_all_agents.append(np.sum(time_optimization_at_each_slot_takes))
                            
        # reward_log.append(np.average(record_reward[(i_episode*n_step_per_episode) : (i_episode+1)*n_step_per_episode ] )) ##  log for average rewards for 100 time steps and all episodes=n_episode 
        return_per_episode=1* np.sum( record_reward[(i_episode*n_step_per_episode) : (i_episode+1)*n_step_per_episode ] ) ##  log for (R)eturn= discount* sumo_rewards) over for 100 time steps and all episodes=n_episode
        return_log.append(return_per_episode) ##  log for (R)eturn= discount* sumo_rewards) over for 100 time steps and all episodes=n_episode
        print("return of 100 steps: ",  return_per_episode)
    print('Training Done. Saving models...')
    
    for i in range(n_veh):
       for j in range(n_neighbor):
           model_path = label + '/agent_' + str(i * n_neighbor + j)
           save_models(sesses[i * n_neighbor + j], model_path)


    current_dir = os.getcwd()
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_4000.mat')
    scipy.io.savemat(reward_path, {'reward_4000': record_reward})
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/return_log_4000.mat')
    scipy.io.savemat(reward_path, {'return_log_4000': return_log})
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/reliability_list_4000.mat')
    scipy.io.savemat(reward_path, {'reliability_list_4000':np.array(env.reliability_list).reshape(-1,1)})
    
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_check_4000.mat')
    scipy.io.savemat(reward_path, {'reward_check_4000': np.array(env.reward_check).reshape(-1,1,n_veh) })
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/strategy_time_all_agents_per_step.mat')
    scipy.io.savemat(reward_path, {'strategy_time_all_agents_per_step':np.array(strategy_time_all_agents_per_step).reshape(-1)})
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/time_optimization_at_each_slot_all_agents.mat')
    scipy.io.savemat(reward_path, {'time_optimization_at_each_slot_all_agents':np.array(time_optimization_at_each_slot_all_agents).reshape(-1)})
    
    

    record_loss = np.asarray(record_loss).reshape((-1, n_veh*n_neighbor))  ## 4 loss functions: for each agent
    loss_path = os.path.join(current_dir, "model/" + label + '/train_loss_4000.mat')
    scipy.io.savemat(loss_path, {'train_loss_4000': record_loss})
    
    
    with open("time_and_parameters.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('Number of trainable parameters per agent: ' + str(total_parameters_per_agent) + '\n')
        f.write('Avg. strategy_time_all_agents_per_step: ' + str(round(np.average(strategy_time_all_agents_per_step), 5)) + 'sec \n')
        f.write('Avg. time_optimization_at_each_slot_all_agents: ' + str(round(np.average(time_optimization_at_each_slot_all_agents), 5)) + 'sec \n')
    
####################################################################################################



# -------------- Testing --------------
####################################################################################################
def get_state_test(env, idx=(0,0), ind_episode=1, epsi=0.02):
    """ Get state from the environment """
    
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :])
    
    V2I_fast_names =[ 'V2I_channel_(strongest)', 'V2I_channel_((2nd strongest)', 'V2I_channel_(3rd strongest)', 'V2I_channel_(weakest)']
    V2I_fast_dict= env.create_state_space_dictionary( V2I_fast_names, np.sort(   V2I_fast.reshape(-1))[::-1] )
    sorted_V2I= dict(sorted(  V2I_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2I)
    
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] )
    V2V_fast_names =['V2V_channel_(strongest)', 'V2V_channel_(2nd strongest)', 'V2V_channel_(3rd strongest)','V2V_channel_(4th strongest)',\
                      'V2V_channel_(5th strongest)','V2V_channel_(6th strongest)',\
                          'V2V_channel_(7th strongest)','V2V_channel_(8th strongest)','V2V_channel_(9th strongest)',\
                              'V2V_channel_(10th strongest)','V2V_channel_(11th strongest)' ,'V2V_channel_(12th strongest)','V2V_channel_(13th strongest)',\
                                'V2V_channel_(14th strongest)','V2V_channel_(15th strongest)', 'V2V_channel_(weakest)']
    V2V_fast_dict= env.create_state_space_dictionary( V2V_fast_names, np.sort(   V2V_fast.reshape(-1))[::-1] )
    sorted_V2V= dict(sorted(  V2V_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2V)
    
    
    
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :])
    V2V_intrfr_names =['Interference power_(strongest)', 'Interference power_(2nd strongest)', 'Interference power_(3rd strongest)', 'Interference power_(weakest)']
    V2V_intrfr_dict= env.create_state_space_dictionary( V2V_intrfr_names, np.sort(    V2V_interference.reshape(-1))[::-1] )
    sorted_V2V_intrfr= dict(sorted(  V2V_intrfr_dict.items(), key=lambda item: item[1], reverse=True))
    
    combined_state = {}
    combined_state.update(sorted_V2I)
    combined_state.update(sorted_V2V)
    combined_state.update(sorted_V2V_intrfr)

    return combined_state, np.array(list(combined_state.values()))


# from collections import defaultdict
new_dict = defaultdict(list)


state_log_test=np.zeros([n_episode_test, n_step_per_episode, n_veh,n_input])
joint_action_log_test=np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
# power_log=np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
# spectrum_log=np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])

if IS_TEST:
    print("\nRestoring the model...")
    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = label + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses[i * n_neighbor + j], model_path)
            
    record_test_reward = np.zeros([n_episode_test*n_step_per_episode, 1])
    V2I_rate_list = []
    V2V_success_list = []
    V2I_rate_list_rand = []
    V2V_success_list_rand = []
    metric_list=[]
    metric_rand_list= []
    
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    efective_rate_marl=np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    V2I_rate_marl=  np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    efective_rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    V2I_rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    
    rate_full = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    efective_rate_full = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    V2I_rate_full = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    
    
    
    power_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        V2I_rate_per_episode_full = []
        metric_per_episode=[]
        metric_per_episode_rand=[]
        for test_step in range(n_step_per_episode):
            time_step = idx_episode*n_step_per_episode + test_step
            # trained models
            action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    # state_old = get_state(env, [i, j], 1, epsi_final)
                    combined_state, state_old = get_state_test(env, [i, j])

                    action = predict(sesses[i*n_neighbor+j], state_old, epsi_final/100, True)
                    action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                    action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level
                    
                    for key, value in combined_state.items():
                        new_dict[key].append(value)
                    state_data_dict = dict(new_dict)
                    state_dataframe=pd.DataFrame(state_data_dict)
                    
                    state_log_test[idx_episode, test_step, i, :]= state_old ## array of states per agent j=1 , i varies from 0-3
                    joint_action_log_test[idx_episode, test_step, i, j] = action  ### value between 0-15 for 16 different combination of actions
                    # power_log[idx_episode, test_step, i, j] = env.V2V_power_dB_List[int( action_all_testing[i, j, 1])]
                    # spectrum_log[idx_episode, test_step, i, j] = int(action_all_testing[i, j, 0])

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate,  effec_rate, test_reward = env.act_for_testing(action_temp)
            record_test_reward[time_step]= test_reward
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps
            metric_per_episode.append(env.metric)
            metric_per_episode_rand.append(env.metric_rand)
            ####### elemenst to use for CDF plotting#############
            rate_marl[idx_episode, test_step,:,:] = V2V_rate
            efective_rate_marl[idx_episode, test_step,:,:] = effec_rate
            V2I_rate_marl[idx_episode, test_step,:,:] = V2I_rate.reshape(effec_rate.shape)


            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor]) # power
            V2I_rate_random, V2V_success_random, V2V_rate_random,  effec_rate_random = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_random))  # sum V2I rate in bps
            ####### elemenst to use for CDF plotting random #############
            rate_rand[idx_episode, test_step, :, :] = V2V_rate_random
            efective_rate_rand[idx_episode, test_step, :, :] = effec_rate_random
            V2I_rate_rand[idx_episode, test_step, :, :] =  V2I_rate_random.reshape(effec_rate_random.shape)
            

            
            # full power baseline
            action_full = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_full[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor]) # band
            action_full[:, :, 1] = len(env.V2V_power_dB_List) - 1  # Maximum power index
            V2I_rate_full_power, V2V_success_full, V2V_rate_full,  effec_rate_full_power = env.act_for_testing_rand(action_full)
            V2I_rate_per_episode_full.append(np.sum(V2I_rate_full_power))  # sum V2I rate in bps
            ####### elemenst to use for CDF plotting full power #############
            rate_full[idx_episode, test_step, :, :] = V2V_rate_full
            efective_rate_full[idx_episode, test_step, :, :] = effec_rate_full_power
            V2I_rate_full[idx_episode, test_step, :, :] =  V2I_rate_full_power.reshape(effec_rate_full_power.shape)
            

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_random)
  
                
        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        V2I_rate_list_rand.append(np.mean(V2I_rate_per_episode_rand))
        
        metric_list.append(np.mean(metric_per_episode))  # metrics for XAI list
        metric_rand_list.append(np.mean(metric_per_episode_rand))  # metrics for XAI list

        print('V2I', round(np.average(V2I_rate_per_episode), 3), 'rand', round(np.average(V2I_rate_per_episode_rand), 3))
        print('V2V_succes', V2V_success_list[idx_episode], 'rand', V2V_success_list_rand[idx_episode])
        print('metric', round(np.average(metric_per_episode), 3), 'metric_rand',  round(np.average(metric_per_episode_rand), 3) )

        

    print('-------- marl -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list), 4))
    print('Metric:', round(np.average(metric_list), 4))
    
    print('-------- random -------------')
    print('n_veh:', n_veh, ', n_neighbor:', n_neighbor)
    print('Sum V2I rate:', round(np.average(V2I_rate_list_rand), 2), 'Mbps')
    print('Pr(V2V success):', round(np.average(V2V_success_list_rand), 4))
    print('Metric:', round(np.average(metric_rand_list), 4))

    with open("Data.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('n_veh: ' + str(n_veh) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum V2I rate: ' + str(round(np.average(V2I_rate_list), 5)) + ' Mbps\n')
        f.write('Pr(V2V): ' + str(round(np.average(V2V_success_list), 5)) + '\n')
        f.write('Metric: ' + str(round(np.average(metric_list), 5)) + '\n')
        f.write('--------random ------------\n')
        f.write('Rand Sum V2I rate: ' + str(round(np.average(V2I_rate_list_rand), 5)) + ' Mbps\n')
        f.write('Rand Pr(V2V): ' + str(round(np.average(V2V_success_list_rand), 5)) + '\n')
        f.write('Rand Metric: ' + str(round(np.average(metric_rand_list), 5)) + '\n')


   # Specify the folder where the files will be saved
    folder_path = r'C:\Users\nkhan20\Desktop\XAI_Simplified\model\marl_model_Original\testing_for_cdf_files'
    
    # Save reward_test data
    reward_path = os.path.join(folder_path, 'reward_test.mat')
    scipy.io.savemat(reward_path, {'reward_test': record_test_reward})
    
    # Save rate_marl data
    marl_path = os.path.join(folder_path, 'rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    
    # Save  efective_rate_marl
    efective_rate_marl_path = os.path.join(folder_path, 'efective_rate_marl.mat')
    scipy.io.savemat(efective_rate_marl_path, {'efective_rate_marl':  efective_rate_marl})
    
    # Save V2I_rate_marl data
    V2I_rate_marl_path = os.path.join(folder_path, 'V2I_rate_marl.mat')
    scipy.io.savemat(V2I_rate_marl_path, {'V2I_rate_marl': V2I_rate_marl})
    
    # Save rate_rand data
    rand_path = os.path.join(folder_path, 'rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})
    
    # Save efective_rate_rand data
    efective_rate_rand_path = os.path.join(folder_path, 'efective_rate_rand.mat')
    scipy.io.savemat(efective_rate_rand_path, {'efective_rate_rand': efective_rate_rand})
    
    # Save V2I_rate_rand data
    V2I_rate_rand_path = os.path.join(folder_path,'V2I_rate_rand.mat')
    scipy.io.savemat(V2I_rate_rand_path, {'V2I_rate_rand':  V2I_rate_rand})
    

    # Save V2I_rate_full data
    V2I_rate_full_path = os.path.join(folder_path, 'V2I_rate_full.mat')
    scipy.io.savemat(V2I_rate_full_path, {'V2I_rate_full': V2I_rate_full})
    
    # Save rate_full data
    rate_full_path = os.path.join(folder_path, 'rate_full.mat')
    scipy.io.savemat(rate_full_path, {'rate_full': rate_full})
    
    # Save efective_rate_full data
    efective_rate_full_path = os.path.join(folder_path, 'efective_rate_full.mat')
    scipy.io.savemat(efective_rate_full_path, {'efective_rate_full': efective_rate_full})
    
    # Save reliability_marl data
    reliability_marl_path = os.path.join(folder_path, 'reliability_marl.mat')
    scipy.io.savemat(reliability_marl_path, {'reliability_marl': V2V_success_list})
    
    # Save reliability_rand data
    reliability_rand_path = os.path.join(folder_path, 'reliability_rand.mat')
    scipy.io.savemat(reliability_rand_path, {'reliability_rand': V2V_success_list_rand})


    
    ###############################################################
    #####################  SHAP and XAI  ##########################
    #init the JS visualization code
    shap.initjs()

    state_dataframe=pd.DataFrame(state_data_dict)
    feature_names_list =state_dataframe.columns.tolist()  ## use this to get names from pd frame
    
    # background=state_log
    
    test_size_sampler=1000
    Shap_values=[]
    X_test=[] ## same as D^shap in algorithm. seperate dataset for each agent
    
    SHAP_train= state_log_test.reshape(n_episode_test*n_step_per_episode,n_veh,n_input)
    SHAP_action=joint_action_log_test.reshape(n_episode_test*n_step_per_episode,n_veh,1)
    
    for i in range(n_veh):     
        for j in range(n_neighbor):
            SHAP_x= SHAP_train[:,i,:] ## SHAP matrix for each vehicle  [samples x number of features]

            X_test.append(SHAP_x[:test_size_sampler])
            background =  SHAP_x[np.random.choice( SHAP_x.shape[0], 3000, replace=False)]
            
            x_tensor = x  
            y_tensor = y 
            model= (x_tensor,y_tensor)
            
            explainer = shap.DeepExplainer(model, background,sesses[i*n_neighbor+j] ) 
            print("#########################")
            print("stuck after this")
            print("#########################")
            Shap_values.append(explainer.shap_values(X_test[i]))  
    for i in range(n_veh):                                                           
 
            plt.rcParams.update({'font.size': 16})
            plt.figure(figsize=(10, 8))  # Adjust the width and height as needed
               
            shap.summary_plot( np.mean(np.abs(np.array( Shap_values[i] )),axis= 1),  plot_type = 'bar',max_display=1000, feature_names = feature_names_list, color= 'mediumorchid')  
            plt.xticks(fontsize=16)  # Adjust the font size as needed
            plt.yticks(fontsize=16)  # Adjust the font size as needed
            file_name = "Global_explainer_agent_" + str(i) + ".pdf" 
            path_to_save = r'C:\Users\nkhan20\Desktop\XAI_Simplified\XAI_plots\\' + file_name
            plt.savefig(path_to_save,bbox_inches='tight',dpi=1000 )
            plt.show()
            # plt.savefig( file_name,bbox_inches='tight',dpi=1000)

     

    # Calculate feature importance (absolute mean of SHAP values) ## take the mean of the columns(along S) of each matrix (VxCxSxF) is calculated.
    ##The vectors of  mean SHAP values for each class are summed and orderedin a decreasing way.
    mean_abs_shap= np.mean(np.abs(np.array(Shap_values)),axis=2) ## take abs mean across all samples for each class for all vehicles-->result(VxCxF)
    
    SHAP_avg = (np.sum( mean_abs_shap,axis=1 )) ## mean SHAP values for each class are summed  for  all  vehicles -->result(VxF)
    
    feature_importance = np.mean(np.abs(SHAP_avg), axis=0) ## take absolute mean across all vehicle.,i.,e model averaging
    # Rank features by importance                                           ## this is what summary plot does
    feature_ranking = np.argsort(feature_importance)
    
    
    def softmax(x):
           return np.exp(x) / np.sum(np.exp(x))
    ###### apply softmax so for a particular sample, the shap values sum upto one#####
    plt.figure()
    # plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 8))  # Adjust the width and height as needed
    shap.summary_plot(np.apply_along_axis(softmax, axis=1, arr=SHAP_avg), plot_type='bar', max_display=1000, feature_names=feature_names_list)
    plt.title('Averaged global feature importance')
    file_name= "Averaged SHAP_summary.pdf"
    path_to_save = r'C:\Users\nkhan20\Desktop\XAI_Simplified\XAI_plots\Averaged SHAP_summary_plot.pdf'
    plt.savefig(path_to_save,bbox_inches='tight',dpi=1000 )
    plt.show()  
    
    def evaluation_model(X_data):   
        metric_list_simple=[]
        for idx_episode in range(1): ## here we test over a single episode to calculate the reliability only
            
              metric_per_episode_simple=[]
              for test_step in range(test_size_sampler): ## 100 steps to match the size of test dataset
                  # trained models
                  action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
                  for i in range(n_veh):
                      for j in range(n_neighbor):
                          # state_old = get_state(env, [i, j], 1, epsi_final)
                          # combined_state, state_old = get_state_test(env, [i, j])
                          
                          # state_old= np.array(list(combined_state.values()))
                          action = predict(sesses[i*n_neighbor+j],X_data[i,test_step,:], epsi_final/100, True)
                          action_all_testing[i, j, 0] = action % n_RB  # chosen RB
                          action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level
                          
    
                  action_temp = action_all_testing.copy()
                  V2I_rate, V2V_success, V2V_rate,  effec_rate, test_reward = env.act_for_testing(action_temp)
                  metric_per_episode_simple.append(V2V_success)

              metric_list_simple.append(np.mean(metric_per_episode_simple))  # metrics for XAI list
              metric_list_simple.count(1.0)
              # Calculate the percentage of ones
              percentage_ones = (metric_list_simple.count(1) / len(metric_list_simple)) * 100
         
        return metric_per_episode_simple,  percentage_ones,   round(np.average(metric_list_simple), 3)  
    

    
  # Define the function to randomize features
    def randomize_features(X_data, features_to_randomize, noise_scale):
        X_rand_copy = X_data.copy()  # Create a copy of the input array
        n_veh, n_samples, n_features = X_rand_copy.shape
        for i in range(n_veh): 
            for feature_idx in features_to_randomize:
                feature_samples = X_data[i,:, feature_idx]
                # Calculate the standard deviation
                feature_std  = np.std(feature_samples)
                noise= np.random.normal(loc=0, scale=10*feature_std, size=(n_samples,))
                # noise =np.random.uniform(low=-feature_std, high=feature_std, size=(n_samples,))
                # noise= np.random.normal(loc=0, scale=noise_scale, size=(n_samples,))
                X_rand_copy[i,:, feature_idx] += noise
                # X_rand_copy[i,:, feature_idx] = 0 
        return X_rand_copy  # Return the modified test dataset with randomized features


    def evaluate_reliability_difference(alpha_original, alpha_random, precision_threshold):  
    # Evaluate the difference in reliability and compare it with the precision threshold
        
        return abs(alpha_original-alpha_random) >= precision_threshold
 
    # Define the number of least important features to consider
    features_retained=[]
    
    alphas_original_threshold=[]
    alphas_random_threshold=[]
    
    precision_threshold_list = [round(0.01 + i * 0.01, 3) for i in range(10)]
    for p in precision_threshold_list:
        print('----- precision_threshold', p, '-----'+ '\n')
        alpha_original_list=[]
        alpha_random_list=[]
        f = 1  # Example: Consider 1 least important features
        while True:   
            # Randomize the values of the k least important features within the test dataset
            X_test_rand = randomize_features(np.array(X_test), feature_ranking[:f] , noise_scale=1)
            
            # print("\nAre the arrays different?", not np.array_equal(X_test, X_test_rand))
            reliability_list_e_original, percentage_ones_original,  alpha_original= evaluation_model(np.array(X_test))
            reliability_list_e_rand, percentage_ones_rand,  alpha_random= evaluation_model(np.array(X_test_rand))
            
            alpha_original_list.append( alpha_original)
            alpha_random_list.append( alpha_random)
        
            if evaluate_reliability_difference(alpha_original, alpha_random, precision_threshold=p) :
                print("important enough feature randomized")
                print("features to retain is: ", len(state_old) - f) 
                print("original  model metric for reliability:", alpha_original)
                print("simplified model metric for reliability:", alpha_random)
                # print("Percentage reliability original:", percentage_ones_original)
                # print("Percentage reliability random:", percentage_ones_rand)
                break
            else:
                print("not important: updating the f value to: ", f+1 )
                f += 1
        
 
        alphas_original_threshold.append(alpha_original_list)
        alphas_random_threshold.append(alpha_random_list)
        features_retained.append(len(state_old)-f)
        print('----- features_retained', features_retained, '-----')
        
    with open("Precisionvsfeatures.txt", "a") as f:
        f.write('-------- marl, ' + label + '------\n')
        f.write('feature_ranking= np.argsort(feature_importance):' + str(feature_ranking) + '\n')
        f.write('features_retained: ' + str(features_retained) + '\n')
        f.write('precision threshold:' + str(precision_threshold_list) + '\n')
       




