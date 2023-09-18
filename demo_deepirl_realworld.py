import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from collections import namedtuple

from realGrid import real_grid
from realGrid import value_iteration
from trajectory import *
from deep_irl_realworld import *

PARSER = argparse.ArgumentParser(
    description="argument of deep max entropy inverse learning algorithm")
PARSER.add_argument('-g', '--gamma', default=0.9,
                    type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.2,
                    type=float, help='probability of acting randomly')
PARSER.add_argument('-lr', '--learning_rate', default=0.02,
                    type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=100,
                    type=int, help='number of iterations')
PARSER.add_argument('--restore', dest='restore', action='store_true',
                    help='restore graph from existed checkpoint file')
ARGS = PARSER.parse_args()
print(ARGS)

GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
RESTORE = ARGS.restore

# get the transition probability between states
feature_map_file = './data/ns_SF.csv'
feature_map_excel = pd.read_csv(feature_map_file)
states_list = list(feature_map_excel['fnid'])
fnid_idx = {}
idx_fnid = {}
for i in range(len(states_list)):
    fnid_idx.update({states_list[i]: i})
    idx_fnid.update({i: states_list[i]})
grid = real_grid.RealGridWorld(fnid_idx, idx_fnid, 1-ACT_RAND)
p_a = grid.get_transition_mat()
# get feature map of state
feat_map = feature_map_excel.iloc[:, 1:]
index_fnid = feature_map_excel['fnid']
feat_map = np.array(feat_map)

# train the deep-irl
gpd_file = './data/nanshan_grid.shp'
route_file_path = './data/routes_states'

nn_r = DeepIRLFC(feat_map.shape[1], LEARNING_RATE, [
                 0, 0, 0, 0, 0], 40, 30)  # initialize the deep-network

pre_reward= np.zeros((2,3,feat_map.shape[0]),dtype=float)

T0 = time.time()
now_time = datetime.datetime.now()
Step=namedtuple('Step',['state','action'])
print('this training loop start at {}'.format(now_time))
for iteration in range(N_ITERS):
    T1 = time.time()
    this_reward=np.zeros((2,3,feat_map.shape[0]))
    for f in os.listdir(route_file_path):
        genderAge = [0]*5
        gender,age=int(f[0]),int(f[2])
        genderAge[gender], genderAge[age+2] = 1,1
        trajs = np.load(route_file_path+'/'+f,allow_pickle=True)
        trajs = trajs.tolist()
        print("load trajectory done!")
        rewards, nn_r = deepMaxEntIRL2(nn_r, feat_map, p_a, GAMMA, trajs,
                                 LEARNING_RATE, fnid_idx, idx_fnid, gpd_file, genderAge, RESTORE)
        this_reward[gender,age,:]=np.array(rewards).reshape(feat_map.shape[0])

    T2 = time.time()
    print("this iteration lasts {:.2f}s, the loop lasts {:.2f}s, the {}th iteration end at {}".format(
        T2-T1, T2-T0, iteration, datetime.datetime.now()))
    
    reward_difference=np.mean(this_reward-pre_reward)
    print("the current reward difference is {}".format(reward_difference))
    if abs(reward_difference) <= 0.001:
        print('the difference of reward is less than 0.001, then break the loop')
        break
    pre_reward=this_reward  