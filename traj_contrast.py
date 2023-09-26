import numpy as np

from test_dirl_realworld import readFeatureMap
from A_star.A_star_traj import *
from DTW.dtw import DTW
from evaluate_traj_IRL import *

if __name__ == "__main__":
    feat_map_file = './data/ns_sf.csv'
    genderAge=[1,0,1,0,0]

    feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(feat_map_file)
    
    # logit
    reward_logit = getRewardFileRLogit(feat_map,genderAge).tolist()
    reward_logit = [r+np.random.gumbel(0,0.1) for r in reward_logit]
    cost_logit = [1-r for r in reward_logit]
    # ga
    reward_ga = getRewardFileGA(feat_map, genderAge).tolist()
    reward_ga = [r[0]+np.random.gumbel(0,0.08) for r in reward_ga]
    cost_ga = [1-r for r in reward_ga]
    # be 
    tf.reset_default_graph()
    reward_be = getRewardFileBE(feat_map).tolist()
    reward_be = [r[0]+np.random.gumbel(0,0.13) for r in reward_be]
    cost_be = [1-r for r in reward_be]
    # real map
    real_map_logit = Map(states_list, fnid_idx, idx_fnid, cost_logit)
    real_map_ga = Map(states_list, fnid_idx, idx_fnid, cost_ga)
    real_map_be = Map(states_list, fnid_idx, idx_fnid, cost_be)
    
    # True route
    routes_file = './data/routes_states/0_0_states_tuple.npy'
    routes = np.load(routes_file)
    route = np.random.choice(routes)
    route = [r.state for r in route]
    start_fnid, end_fnid = route[0], route[-1]
    print(start_fnid,end_fnid)
    print(route)
    # Generated route
    route_logit = astar(real_map_logit, start_fnid, end_fnid)
    print(route_logit)
    route_ga = astar(real_map_ga, start_fnid, end_fnid)
    print(route_ga)
    route_be = astar(real_map_be, start_fnid, end_fnid)
    print(route_be)