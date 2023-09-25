import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
from collections import namedtuple

import img_utils
from realGrid import real_grid
from realGrid import value_iteration
from trajectory import *
from recursive_logit import *



PARSER = argparse.ArgumentParser(
    description="argument of deep max entropy inverse learning algorithm")
PARSER.add_argument('-g', '--gamma', default=0.9,
                    type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.2,
                    type=float, help='probability of acting randomly')
PARSER.add_argument('-lr', '--learning_rate', default=0.02,
                    type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20,
                    type=int, help='number of iterations')
PARSER.add_argument('--restore', dest='restore', action='store_true',
                    help='restore graph from existed checkpoint file')
ARGS = PARSER.parse_args()


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
feat_map=np.array(feat_map)

route_file_path = './data/routes_states'
Step=namedtuple('Step',['state','action'])
# run trajectory.py,and get the state-action pair of real world trajectory
for f in os.listdir(route_file_path):
    genderAge = [0]*5
    gender,age=int(f[0]),int(f[2])
    genderAge[gender], genderAge[age+2] = 1,1

    trajs = np.load(route_file_path+'/'+f,allow_pickle=True)
    trajs = trajs.tolist()
    rewards_maxent = recursiveLogit(feat_map, genderAge, p_a, GAMMA, trajs, fnid_idx,LEARNING_RATE, N_ITERS)