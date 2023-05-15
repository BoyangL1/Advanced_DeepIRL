import sys
import numpy as np
from test_dirl_realworld import readFeatureMap


def calDistance(x, y):
    return abs(x[0]-y[0])+abs(x[1]-y[1])

def dtw(X, Y):
    l1 = len(X)
    l2 = len(Y)
    D = [[0 for i in range(l1 + 1)] for i in range(l2 + 1)]
    # D[0][0] = 0
    for i in range(1, l1 + 1):
        D[0][i] = sys.maxsize
    for j in range(1, l2 + 1):
        D[j][0] = sys.maxsize
    for j in range(1, l2 + 1):
        for i in range(1, l1 + 1):
            D[j][i] = calDistance(X[i - 1], Y[j-1]) + \
                min(D[j - 1][i], D[j][i - 1], D[j - 1][i - 1])
    return D

def getTraj(policy_save_path,feature_map_file):
    # get real route
    routes = np.load('./data/routes_states.npy')
    route_list=[]
    traj_list=[]
    for route in routes:
        start_fnid = route[0]
        end_fnid = route[-1]
        # read value file
        policys = np.load(policy_save_path)
        _, fnid_idx, _, _ = readFeatureMap(feature_map_file)
        # calculate idx for start and end fnid
        end_fnid_idx = fnid_idx[end_fnid]
        # get determinstic poliy given end fnid
        policy_end = policys[end_fnid_idx]
        traj = [start_fnid]
        while start_fnid != end_fnid:
            start_fnid = policy_end[fnid_idx[start_fnid]]
            traj.append(start_fnid)
        route_list.append(route)
        traj_list.append(traj)
    return route_list,traj_list

def fnidToXY(traj):
    traj_list=[]
    for t in traj:
        x=t%357
        y=(t)//357
        traj_list.append((x,y))
    return traj_list

if __name__ == '__main__':
    policy_save_path = './model/policy_nanshan_determinstic.npy'
    feature_map_file = './data/nanshan_tfidf.xlsx'
    real_traj,pre_traj=getTraj(policy_save_path,feature_map_file)
    real_traj_list=[]
    pre_traj_list=[]
    for r in real_traj:
        real_traj_list.append(fnidToXY(r))
    for t in pre_traj:
        pre_traj_list.append(fnidToXY(t))
    distance=[]
    length=len(real_traj_list)
    for i in range(length):
        dist=dtw(pre_traj_list[i],real_traj_list[i])
        distance.append(dist[-1][-1])
    print(distance)
    print("the average distance is {}".format(np.mean(distance)))