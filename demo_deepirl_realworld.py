import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from collections import namedtuple

import img_utils
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
feature_map_file = './data/nanshan_tfidf.xlsx'
feature_map_excel = pd.read_excel(feature_map_file)
states_list = list(feature_map_excel['fnid'])
fnid_idx = {}
idx_fnid = {}
for i in range(len(states_list)):
    fnid_idx.update({states_list[i]: i})
    idx_fnid.update({i: states_list[i]})
grid = real_grid.RealGridWorld(fnid_idx, idx_fnid, 1-ACT_RAND)
p_a = grid.get_transition_mat()
# get feature map of state
feature_map1 = feature_map_excel.iloc[:, 3:12]
feature_map2 = feature_map_excel.iloc[:, 14:35]
index_fnid = feature_map_excel['fnid']
feat_map = pd.concat([feature_map1, feature_map2], axis=1)
feat_map = np.array(feat_map)

# train the deep-irl
gpd_file = './data/nanshan.shp'
route_file_path = './data/routes_states'

nn_r = DeepIRLFC(feat_map.shape[1], LEARNING_RATE, [0,0,0,0,0], 40, 30) # initialize the deep-network
for iteration in range(N_ITERS):
    for f in os.listdir(route_file_path):
        genderAge = [0]*5
        genderAge[int(f[0])], genderAge[int(f[2])] = 1, 1
        trajs=[]
        routes_states = np.load(route_file_path+'/'+f)
        for route_state in routes_states:
            sta_act = getActionOfStates(route_state)
            trajs.append(sta_act)
        rewards = deepMaxEntIRL2(nn_r,feat_map, p_a, GAMMA, trajs,
                                 LEARNING_RATE, fnid_idx, idx_fnid, gpd_file, genderAge, RESTORE)
        
# img_utils.rewardVisual(rewards, idx_fnid, gpd_file,"recovered reward map")
# plt.savefig('./img/reward_final.png')
# plt.show()
# plt.close()