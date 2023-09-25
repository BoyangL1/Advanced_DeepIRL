import numpy as np
import argparse
import pandas as pd
from collections import namedtuple

from realGrid import real_grid
from deep_irl_be import *

PARSER = argparse.ArgumentParser(
    description="argument of deep max entropy inverse learning algorithm")
PARSER.add_argument('-g', '--gamma', default=0.9,
                    type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.2,
                    type=float, help='probability of acting randomly')
PARSER.add_argument('-lr', '--learning_rate', default=0.02,
                    type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=10,
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
Step=namedtuple('Step',['state','action'])

# get the transition probability between states
feature_map_file = './data/nanshan_tfidf_be.csv'
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

# train the deep-irl without built environment
gpd_file = './data/nanshan_grid.shp'
route_file_path = './data/routes_states'

trajs = []
for f in os.listdir(route_file_path):
    traj = np.load(route_file_path+'/'+f,allow_pickle=True)
    traj = traj.tolist()
    trajs.extend(traj)

rewards = deepMaxEntIRL(feat_map, p_a, GAMMA, trajs,
                        LEARNING_RATE, N_ITERS, fnid_idx, idx_fnid, gpd_file, RESTORE)