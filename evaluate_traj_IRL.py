import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from deep_irl_realworld import *
import deep_irl_be
from test_dirl_realworld import readFeatureMap
from A_star.A_star_traj import *
from DTW.dtw import DTW

def getRewardFileGA(feat_map, genderAge):
    """
    get reward of feature map

    Args:
        feat_map (N*M:2d_matrix): feature map of zone

    Returns:
        rewards (N*1): reward of each grid
    """
    nn_r = DeepIRLFC(feat_map.shape[1], 0.02,genderAge, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return

    oh = [genderAge for _ in range(feat_map.shape[0])]
    rewards = normalize(nn_r.get_rewards(feat_map, oh))
    return rewards

def getRewardFileBE(feat_map):
    nn_r = deep_irl_be.DeepIRLFC(feat_map.shape[1], 0.02,40, 30)
    model_file = './model/be'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return

    rewards = normalize(nn_r.get_rewards(feat_map))
    return rewards

def getRewardFileRLogit(feat_map,genderAge):
    genderAge=np.array(genderAge)
    genderAge = np.tile(genderAge, (feat_map.shape[0], 1))
    feat_map = np.concatenate((genderAge, feat_map), axis=1)

    theta = np.load('./model/rlogit_realworld.npy')
    rewards = normalize(np.dot(feat_map, theta))
    return rewards

def trajEvaluate(routes_file,real_map,gpd_file,save_picture=False):
    routes = np.load(routes_file)
    all_length=0
    list_length=[]
    for i,route in enumerate(routes):
        route = [r.state for r in route]
        # generate trajectory
        start_fnid=route[0]
        end_fnid = route[-1]
        route_generate = astar(real_map, start_fnid, end_fnid)
        # calculate dtw distance between trajectory
        traj_dtw=DTW(route,route_generate)
        D=traj_dtw.dtw()
        length=(len(route_generate)+len(route))/2
        dtw_length=D[-1][-1]/length
        # save figure
        if save_picture:
            trajContrast(route, real_map, gpd_file, False)
            plt.title('DTW Length is {}'.format(dtw_length))
            plt.savefig('./img/traj_distance/{}.png'.format(i))
        print(dtw_length)
        all_length+=dtw_length
        list_length.append(dtw_length)

    avg_length=all_length/len(routes)
    return avg_length,list_length

if __name__ == "__main__":
    feat_map_file = './data/ns_sf.csv'
    gpd_file = './data/nanshan_grid.shp'
    routes_file = './data/routes_states/0_0_states_tuple.npy'
    genderAge=[1,0,1,0,0]

    feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(feat_map_file)

    # NOTE: get different reward
    # rewards = getRewardFileGA(feat_map, genderAge).tolist()
    # rewards = getRewardFileBE(feat_map).tolist()
    # cost = [1-r[0] for r in rewards]
    
    rewards = getRewardFileRLogit(feat_map,genderAge).tolist()
    cost = [1-r for r in rewards]

    real_map = Map(states_list, fnid_idx, idx_fnid, cost*10)

    # trajectory evaluation
    avg_length,list_length=trajEvaluate(routes_file,real_map,gpd_file)

    np.save('./data/evaluate/traj_evaluate_rl.npy',list_length)