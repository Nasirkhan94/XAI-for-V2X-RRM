from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
from math import log, e

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import Environment_marl_shannon
from replay_memory import ReplayMemory
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

label = 'sarl_model_shannon'

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl_shannon.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 4000## originally 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.6*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4 

n_episode_test = 100# test episodes  

######################################################

def get_first_n_items(dictionary, n):
        return dict(list(dictionary.items())[:n])


def get_state(env, idx=(0,0) ):
    """ Get state from the environment """
        
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - 80)/60 
    
    V2I_fast_names =[ 'V2I_channel_(strongest)', 'V2I_channel_((2nd strongest)', 'V2I_channel_(3rd strongest)', 'V2I_channel_(weakest)']
    V2I_fast_dict= env.create_state_space_dictionary( V2I_fast_names, np.sort(   V2I_fast.reshape(-1))[::-1] )
    sorted_V2I= dict(sorted(  V2I_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2I)
    
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80)/60 
    V2V_fast_names =['V2V_channel_(Strongest)', 'V2V_channel_(2nd strongest)', 'V2V_channel_(3rd strongest)','V2V_channel_(4th strongest)',\
                      'V2V_channel_(5th strongest)','V2V_channel_(6th strongest)',\
                          'V2V_channel_(7th strongest)','V2V_channel_(8th strongest)','V2V_channel_(9th strongest)',\
                              'V2V_channel_(10th strongest)','V2V_channel_(11th strongest)' ,'V2V_channel_(12th strongest)','V2V_channel_(13th strongest)',\
                                'V2V_channel_(14th strongest)','V2V_channel_(15th strongest)', 'V2V_channel_(weakest)']
    V2V_fast_dict= env.create_state_space_dictionary( V2V_fast_names, np.sort(   V2V_fast.reshape(-1))[::-1] )
    sorted_V2V= dict(sorted(  V2V_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2V)
  
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    V2V_intrfr_names =['Interference power_(strongest)', 'Interference power_(2nd strongest)', 'Interference power_(3rd strongest)', 'Interference power_(weakest)']
    V2V_intrfr_dict= env.create_state_space_dictionary( V2V_intrfr_names, np.sort(    V2V_interference.reshape(-1))[::-1] )
    sorted_V2V_intrfr= dict(sorted(  V2V_intrfr_dict.items(), key=lambda item: item[1], reverse=True))
    
    combined_state = {}
    combined_state.update(sorted_V2I)
    combined_state.update(sorted_V2V)
    combined_state.update(sorted_V2V_intrfr)
 
    # return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs))
    return  np.array(list(combined_state.values()))
    
##########################################################################
# -----------------------------------------------------------

# -----------------------------------------------------------
# n_hidden_1 = 500
# n_hidden_2 = 250
# n_hidden_3 = 120
# n_input = len(get_state(env=env))
# n_output = n_RB * len(env.V2V_power_dB_List)

n_hidden_1 = len(get_state(env))*5
n_hidden_2 = len(get_state(env))*3
n_hidden_3 = len(get_state(env))*2
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
    y = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
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
    saver = tf.train.Saver()


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

    current_dir = os.path.dirname(os.path.realpath(__file__))
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
print("Initializing agent...")
agent = Agent(memory_entry_size=len(get_state(env)))

sess = tf.Session(graph=g,config=my_config)
sess.run(init)

# ------------------------- Training -----------------------------
if IS_TRAIN:
    
    # reward_log=[]
    return_log=[]
    record_loss = []
    
    record_reward_shannon = np.zeros([n_episode * n_step_per_episode, 1])
    record_reward_rand = np.zeros([n_episode * n_step_per_episode, 1])
    record_reward_full= np.zeros([n_episode * n_step_per_episode, 1])
    record_loss = []
    action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    action_all_training_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    action_all_training_full = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    time_step = 0
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final/100
        if i_episode%100 == 0:
            env.renew_positions() # update vehicle position
            env.renew_neighbor()
            env.renew_channel() # update channel slow fading
            env.renew_channels_fastfading() # update channel fast fading

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        for i_step in range(n_step_per_episode):
            time_step = i_episode * n_step_per_episode + i_step

            remainder = i_step % (n_veh * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            state = get_state(env, [i, j] )
            action = predict(sess, state, epsi)
            action_all_training[i, j, 0] = action % n_RB  # chosen RB
            action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level
            
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            record_reward_shannon[time_step] = train_reward
            
            ######## compure training performance with random/max power action ####################
            
            # random baseline
           
            action_all_training_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
            action_all_training_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor])  # power
            
            action_temp_rand = action_all_training_rand.copy()
            train_reward_rand = env.act_for_training(action_temp_rand)
            record_reward_rand[time_step] = train_reward_rand
            
            
            ######## compure training performance with Full power action ####################
            # random baseline
           
            action_all_training_full[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
            action_all_training_full[:, :, 1] = len(env.V2V_power_dB_List) - 1  # Maximum power index
            
            action_temp_full = action_all_training_full.copy()
            train_reward_full = env.act_for_training(action_temp_full)
            record_reward_full[time_step] = train_reward_full
            
            ################################################################################

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            state_new = get_state(env, [i, j],)
            agent.memory.add(state, state_new, train_reward, action)  # add entry to this agent's memory

            # training this agent
            if time_step % mini_batch_step == mini_batch_step - 1:
                loss_val_batch = q_learning_mini_batch(agent, sess)
                record_loss.append(loss_val_batch)
                print('step:', time_step, 'loss', loss_val_batch)
            if time_step % target_update_step == target_update_step - 1:
                update_target_q_network(sess)
                print('Update target Q network...')
        
        
        # reward_log.append(np.average(record_reward[(i_episode*n_step_per_episode) : (i_episode+1)*n_step_per_episode ] )) ##  log for average rewards for 100 time steps and all episodes=n_episode 
        return_per_episode=1* np.sum( record_reward_shannon[(i_episode*n_step_per_episode) : (i_episode+1)*n_step_per_episode ] ) ##  log for (R)eturn= discount* sumo_rewards) over for 100 time steps and all episodes=n_episode
        return_log.append(return_per_episode) ##  log for (R)eturn= discount* sumo_rewards) over for 100 time steps and all episodes=n_episode
        print("return of 100 steps: ",  return_per_episode)

    print('Training Done. Saving models...')
    model_path = label + '/agent'
    save_models(sess, model_path)

    current_dir = os.getcwd()
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_sarl.mat')
    scipy.io.savemat(reward_path, {'reward_sarl': record_reward_shannon})
    
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_rand.mat')
    scipy.io.savemat(reward_path, {'reward_rand': record_reward_rand})
    
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_full.mat')
    scipy.io.savemat(reward_path, {'reward_full': record_reward_full})
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/return_log_4000.mat')
    scipy.io.savemat(reward_path, {'return_log_4000': return_log})
    
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/reliability_list_4000.mat')
    scipy.io.savemat(reward_path, {'reliability_list_4000':np.array(env.reliability_list).reshape(-1,1)})
    
    reward_path = os.path.join(current_dir, "model/" + label + '/reward_check_4000.mat')
    scipy.io.savemat(reward_path, {'reward_check_4000': np.array(env.reward_check).reshape(-1,1,n_veh) })
    
    
    record_loss = np.asarray(record_loss).reshape((-1, n_veh*n_neighbor))  ## 4 loss functions: for each agent
    loss_path = os.path.join(current_dir, "model/" + label + '/train_loss_4000.mat')
    scipy.io.savemat(loss_path, {'train_loss_4000': record_loss})


# -------------- Testing --------------
####################################################################################################
def get_state_test(env, idx=(0,0), ind_episode=1, epsi=0.02):
    """ Get state from the environment """
    
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - 80)/60 
    
    V2I_fast_names =[ 'V2I_channel_(strongest)', 'V2I_channel_((2nd strongest)', 'V2I_channel_(3rd strongest)', 'V2I_channel_(weakest)']
    V2I_fast_dict= env.create_state_space_dictionary( V2I_fast_names, np.sort(   V2I_fast.reshape(-1))[::-1] )
    sorted_V2I= dict(sorted(  V2I_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2I)

    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80)/60 
    V2V_fast_names =['V2V_channel_(Strongest)', 'V2V_channel_(2nd strongest)', 'V2V_channel_(3rd strongest)','V2V_channel_(4th strongest)',\
                      'V2V_channel_(5th strongest)','V2V_channel_(6th strongest)',\
                          'V2V_channel_(7th strongest)','V2V_channel_(8th strongest)','V2V_channel_(9th strongest)',\
                              'V2V_channel_(10th strongest)','V2V_channel_(11th strongest)' ,'V2V_channel_(12th strongest)','V2V_channel_(13th strongest)',\
                                'V2V_channel_(14th strongest)','V2V_channel_(15th strongest)', 'V2V_channel_(weakest)']
    V2V_fast_dict= env.create_state_space_dictionary( V2V_fast_names, np.sort(   V2V_fast.reshape(-1))[::-1] )
    sorted_V2V= dict(sorted(  V2V_fast_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_V2V)
        
    
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
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


# -------------- Testing --------------
if IS_TEST:
    print("\nRestoring the model...")
    model_path = label + '/agent'
    load_models(sess, model_path)

    V2I_rate_list = []
    V2V_success_list = []
    V2I_rate_list_rand = []
    V2V_success_list_rand = []
    metric_list=[]
    metric_rand_list= []
    
    rate_marl = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])
    power_rand = np.zeros([n_episode_test, n_step_per_episode, n_veh, n_neighbor])

    action_all_testing = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.renew_positions()
        env.renew_neighbor()
        env.renew_channel()
        env.renew_channels_fastfading()

        
        V2I_rate_per_episode = []
        V2I_rate_per_episode_rand = []
        metric_per_episode=[]
        metric_per_episode_rand=[]
        
        for test_step in range(n_step_per_episode):

            remainder = test_step % (n_veh * n_neighbor)
            i = int(np.floor(remainder / n_neighbor))
            j = remainder % n_neighbor
            combined_state, state_old = get_state_test(env, [i, j])
            action = predict(sess, state_old, epsi_final, True)
            action_all_testing[i, j, 0] = action % n_RB  # chosen RB
            action_all_testing[i, j, 1] = int(np.floor(action / n_RB))  # power level

            action_temp = action_all_testing.copy()
            V2I_rate, V2V_success, V2V_rate,  effec_rate = env.act_for_testing(action_temp)
            V2I_rate_per_episode.append(np.sum(V2I_rate))  # sum V2I rate in bps
            metric_per_episode.append(env.metric)
            metric_per_episode_rand.append(env.metric_rand)

            # random baseline
            action_rand = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_veh, n_neighbor])  # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.V2V_power_dB_List), [n_veh, n_neighbor])  # power

            V2I_rate_rand, V2V_success_rand, V2V_rate_rand,  effec_rate_rand = env.act_for_testing_rand(action_rand)
            V2I_rate_per_episode_rand.append(np.sum(V2I_rate_rand))  # sum V2I rate in bps
            rate_rand[idx_episode, test_step, :, :] = V2V_rate_rand


            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            if test_step == int(n_step_per_episode / (n_veh * n_neighbor)) - 1:
                V2V_success_list.append(V2V_success)
                V2V_success_list_rand.append(V2V_success_rand)

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

    current_dir = os.getcwd()
    marl_path = os.path.join(current_dir, "model/" + label + '/rate_marl.mat')
    scipy.io.savemat(marl_path, {'rate_marl': rate_marl})
    
    current_dir = os.getcwd()
    rand_path = os.path.join(current_dir, "model/" + label + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})
    
    current_dir = os.getcwd()
    reliability_marl_path = os.path.join(current_dir, "model/" + label + '/reliability_marl.mat')
    scipy.io.savemat(reliability_marl_path, {'reliability_marl': V2V_success_list})
    
    current_dir = os.getcwd()
    reliability_rand_path = os.path.join(current_dir, "model/" + label + '/reliability_rand.mat')
    scipy.io.savemat( reliability_rand_path, {'reliability_rand': V2V_success_list_rand})

    current_dir = os.getcwd()
    power_rand_path = os.path.join(current_dir, "model/" + label + '/power_rand.mat')
    scipy.io.savemat(power_rand_path, {'power_rand': power_rand})
    


# close sessions
sess.close()

def smooth( Episode_Return,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(Episode_Return,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(Episode_Return[:WSZ-1])[::2]/r
    stop = (np.cumsum(Episode_Return[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def plot_training_rewards( rewards):
          # Calculate a moving average of rewards
          window_size = 1000
          moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards) - window_size + 1)]
     
          # Create a time axis for x-axis
          episodes = range(len(rewards))
     
          # Plot the rewards and moving average
          plt.figure(figsize=(10, 6))
          plt.plot(episodes, smooth(rewards,105), label='Training Rewards', alpha=0.4)
          plt.plot(episodes[window_size - 1:], moving_avg, label=f'Moving Average (window={window_size})', color='r')
          plt.xlabel('Training step')
          plt.ylabel('Reward per time step ')
          plt.legend()
          # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
          # plt.title('Training Rewards in DRL')
          plt.grid(True)
          plt.show()
          plt.savefig('./figures/_%s_%d'%('training_plot',n_veh)+'.pdf', format='pdf', dpi=1000)
         
def plot_training_reliability( rewards):
          # Calculate a moving average of rewards
          window_size = 1000
          moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards) - window_size + 1)]
     
          # Create a time axis for x-axis
          episodes = range(len(rewards))
     
          # Plot the rewards and moving average
          # plt.figure(figsize=(10, 6))
          plt.plot(episodes, smooth(rewards,105), label='Training Rewards', alpha=0.4)
          plt.plot(episodes[window_size - 1:], moving_avg, label=f'Moving Average (window={window_size})', color='r')
          # plt.axhline(y=1e-2, color='k', linestyle='-', label='$\epsilon_{o}=10^{-5}$')
          plt.xlabel('Training step')
          # plt.ylabel('Network Reliability (1-max $\varepsilon_{k}$)')
          plt.ylabel('Network Reliability (1-max $\{\\varepsilon_{k} \}$)')
          # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
          plt.legend()
          # plt.title('Training Rewards in DRL')
          plt.grid(True)
          plt.show()
          # plt.savefig('./figures/_%s_%d'%('treliability_plot',n_veh)+'.pdf', format='pdf', dpi=1000)

 



plot_training_rewards(record_reward_shannon.reshape(-1))

plot_training_reliability(env.reliability_list)

plot_training_reliability(env.network_avg_throughput)


# env.plot_training_rewards( record_reward_shannon_shannon.reshape(-1))
# env.plot_training_rewards( return_log)

####################################
##################################

# Increase font size globally
plt.rcParams.update({'font.size': 14})
# Function to plot training reliability
def plot_training_combined(rewards,label, color):
    window_size = 500
    moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards) - window_size + 1)]
    episodes = range(len(rewards))
    # Plot the rewards and moving average
    # plt.plot(episodes, smooth(rewards, 505), label=label, alpha=0.4, color=color)
    plt.plot(episodes[window_size - 1:], moving_avg,label=label, color=color)

# Load the .mat file
mat_data = scipy.io.loadmat('reward_4000.mat')

# Extract data from the loaded .mat file
# Suppose the data you want to plot is stored in a variable named 'data'
data = mat_data['reward_4000']

# Plotting the data
fig_width_inches = 7.16  ## 7.16 inches for two column width
fig_height_inches = 0.8 * fig_width_inches  # Adjust the height as needed

plt.figure(figsize=(fig_width_inches, fig_height_inches))
# Define sharp colors
proposed_color = 'red'  # Green
single_agent_color = 'blue'
random_allocation_color = 'black'
full_power_color = 'lime'  # Dark blue

# Example usage
# plot_training_combined(data.reshape(-1), label='Proposed Multi-agent scheme', color=proposed_color)
plot_training_combined(record_reward_shannon.reshape(-1), label='Ideal(infinite blocklength scheme) ', color=single_agent_color)
# plot_training_combined(record_reward_rand.reshape(-1), label='Random Allocation', color=random_allocation_color)
# plot_training_combined(record_reward_full.reshape(-1), label='Full power', color=full_power_color)

plt.legend()
plt.show()

record_reward_rand

plt.xlabel('Training step')
plt.ylabel('Normalized reward')
plt.legend()
plt.grid(True)

# Use ScalarFormatter for x-axis ticks

plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter( useMathText=True))
plt.gca().xaxis.offsetText.set_fontsize(14)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# Add more y ticks between 0 and 1
plt.yticks(np.arange(0, 1.1, 0.1))
# plt.savefig('./figures/_%s_%d'%('combined_training',n_veh)+'.pdf', format='pdf', dpi=1000)
# plt.savefig('./figures/_%s_%d'%('combined_training',n_veh)+'.eps', format='eps', dpi=1000)
plt.show()