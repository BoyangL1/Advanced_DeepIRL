from deep_irl_realworld import *
import os
import pandas as pd
import geopandas as gpd
from realGrid import real_grid, value_iteration
from collections import deque

ACT_RAND = 0.3


def readFeatureMap(feature_map_file):
    """
    read feature map from csv feature file

    Args:
        feature_map_file : path of feature file

    Returns:
        feat_map: numpy feature map
        fnid_idx:{fnid:idx}
        idx_fnid:{idx:fnid}
    """
    feature_map_excel = pd.read_csv(feature_map_file)
    states_list = list(feature_map_excel['fnid'])
    fnid_idx = {}
    idx_fnid = {}
    for i in range(len(states_list)):
        fnid_idx.update({states_list[i]: i})
        idx_fnid.update({i: states_list[i]})
    states_list = list(feature_map_excel['fnid'])
    # get feature map of state 
    feature_map1 = feature_map_excel.iloc[:, 22:] 
    feature_map2 = feature_map_excel.iloc[:, 1:22]
    feat_map = pd.concat([feature_map1, feature_map2], axis=1)
    feat_map = np.array(feat_map)
    return feat_map, fnid_idx, idx_fnid, states_list


def createTrajFromPolicy(start_fnid, end_fnid, trajs_len, traj_len, feature_map_file, policy_npy, rand_start_end=True):
    """
    create trajectory from policy trained given by start and end fnid

    Args:
        start_fnid : start fnid
        end_fnid : end fnid
        traj_len : length of trajectory
        feature_map_file : path of feature file
        policy_npy : policy npy file
        rand_start_end : if select start and end fnid randomly. Defaults to True.

    Returns:
        trajs: a list of trajectory
    """
    # get transition possibility map
    feature_map_excel = pd.read_excel(feature_map_file)
    feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(
        feature_map_file)
    ACT_RAND = 0.6
    grid = real_grid.RealGridWorld(fnid_idx, idx_fnid, 1-ACT_RAND)

    # set start and end fnid
    if rand_start_end:
        start_fnid = np.random.choice(states_list)
        end_fnid = np.random.choice(states_list)
    else:
        start_fnid, end_fnid = start_fnid, end_fnid
    # load policy and verify it
    policy = np.load(policy_npy)
    trajs = []
    for i in range(trajs_len):
        traj = []
        state = start_fnid
        state_idx = fnid_idx[state]
        action = int(policy[state_idx])
        traj.append(state)
        while True:
            probs = grid.get_transition_states_and_probs(state, action)
            nei_s = [s for s, _ in probs]
            nei_s_prob = [prob for _, prob in probs]
            state = np.random.choice(nei_s, p=nei_s_prob)
            state_idx = fnid_idx[state]
            action = int(policy[state_idx])
            traj.append(state)
            if state == end_fnid or len(traj) >= traj_len:
                break
            if action == 4:
                action = np.random.randint(0, 3)
        trajs.append(traj)
    return trajs


def testRewardFile(feature_map_file,gpd_file,gpd_save_file,img_save_file):
    """
    test reward ckpt file trained by deepirl
    """
    lr = 0.02
    feat_map, _, idx_fnid, states_list = readFeatureMap(feature_map_file)
    nn_r = DeepIRLFC(feat_map.shape[1], lr, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return
    rewards = normalize(nn_r.get_rewards(feat_map))
    # visualize reward through geopandas
    gdf = gpd.read_file(gpd_file)
    gdf['reward'] = 0
    for i in range(len(rewards)):
        fnid = idx_fnid[i]
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = rewards[i]
    gdf.to_file(gpd_save_file, driver='ESRI Shapefile', crs=4326,encoding='utf-8')
    gdf.plot(column='reward', cmap='viridis', legend=False)
    plt.title('reward recovered from ckpt file')
    plt.savefig(img_save_file, dpi=600)
    plt.show()


def testPolicy():
    """
    test policy by generating trajectory artificially
    """
    feature_map_file = './data/nanshan_tfidf.xlsx'
    policy_npy = './model/policy_realworld.npy'
    gpd_file = './data/nanshan.shp'
    start_fnid = 7911
    end_fnid = 9355
    trajs = createTrajFromPolicy(start_fnid, end_fnid, 20, 300,
                                 feature_map_file, policy_npy, False)
    gdf = gpd.read_file(gpd_file)
    gdf['frequency'] = 0
    for traj in trajs:
        for fnid in traj:
            gdf.frequency[gdf.fnid == fnid] += 1
    gdf.plot(column='frequency', cmap='plasma', legend=False)
    plt.title('test trajectory')
    plt.show()


def valueIteration():
    """
    find the optimal value function 
    """
    lr = 0.02
    feature_map_file = './data/nanshan_tfidf.xlsx'
    gpd_file = './data/nanshan.shp'
    # get parameters
    feat_map, fnid_idx, idx_fnid, _ = readFeatureMap(feature_map_file)
    # create graph and restore graph parameters from ckpt file
    nn_r = DeepIRLFC(feat_map.shape[1], lr, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return
    # get reward map of given feature map
    reward_map = normalize(nn_r.get_rewards(feat_map))
    print("Reward map done!")

    # get value map
    grid = real_grid.RealGridWorld(fnid_idx, idx_fnid, 1-ACT_RAND)
    p_a = grid.get_transition_mat()
    N_STATES, _, N_ACTIONS = np.shape(p_a)
    values = np.zeros([N_STATES])
    gamma = 0.9
    error = 0.01
    while True:
        values_tmp = values.copy()

        for s in range(N_STATES):
            v_s = []
            values[s] = max([sum([reward_map[s] + p_a[s, s1, a]*gamma*values_tmp[s1]
                            for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
            # values[s] = max([reward_map[s]+gamma*sum([p_a[s, s1, a]*values_tmp[s1]
            #                 for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

        if max([abs(values[s] - values_tmp[s]) for s in range(N_STATES)]) < error:
            break
    values_np = np.array(values)
    np.save('./model/values_realworld.npy', values_np)
    gdf = gpd.read_file(gpd_file)
    gdf['states'] = 0
    for i in range(len(values_np)):
        fnid = idx_fnid[i]
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = values_np[i]
    gdf.plot(column='states', cmap='viridis', legend=False)
    plt.title('values map')
    plt.show()


def determValueIter():
    """
    determinstic value iteration and get value map and poliy map for every state
    """
    lr = 0.02
    feature_map_file = './data/nanshan_tfidf.xlsx'
    gpd_file = './data/nanshan.shp'
    # get parameters
    feat_map, fnid_idx, idx_fnid, states_list = readFeatureMap(
        feature_map_file)
    # create graph and restore graph parameters from ckpt file
    nn_r = DeepIRLFC(feat_map.shape[1], lr, 40, 30)
    model_file = './model'
    model_name = 'realworld'
    if os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)
    else:
        print("there isn't ckpt file")
        return
    # get reward map of given feature map
    reward_map = normalize(nn_r.get_rewards(feat_map))
    print("Reward map done!")

    # calculate value map given determinstic end fnid
    actions = [0, 1, 2, 3]
    neighbors = [1, -1, -357, 357]
    gamma = 0.9
    value_save_path = './model/values_nanshan_determinstic.npy'
    policy_save_path = './model/policy_nanshan_determinstic.npy'
    stochastic_path = './model/policy_nanshan_stochastic.npy'
    # save determinstic values for every state
    values = []
    policys = []
    for s in states_list:
        value, policy = value_iteration.determinValIteration(
            s, actions, neighbors, gamma, fnid_idx, reward_map)
        values.append(value)
        policys.append(policy)
        print("fnid {} has been finished".format(s))
    values_np = np.array(values)
    policy_np = np.array(policys)
    stochasticPolicy(values, stochastic_path, actions,
                     neighbors, idx_fnid, fnid_idx)
    np.save(value_save_path, values_np)
    np.save(policy_save_path, policy_np)


def stochasticPolicy(values, save_path, actions, neighbors, idx_fnid, fnid_idx):
    """
    generate stochastic policy based on value of each state

    Args:
        values : determintic value map of each state 
        save_path : policy save path
        actions : action list
        neighbors : neighboring action list
        idx_fnid : {idx:fnid}
        fnid_idx : {fnid:idx}
    """
    stochastic_policys = []
    for value in values:
        policy = [[0 for i in range(len(actions))] for i in range(len(value))]
        for idx, val in enumerate(value):
            fnid = idx_fnid[idx]
            for a in actions:
                nei_s = fnid+neighbors[a]
                if nei_s in fnid_idx.keys():
                    policy[idx][a] = value[fnid_idx[nei_s]]
        for i in range(len(policy)):
            sum_value = sum(policy[i])
            policy[i] = list(map(lambda x: x/(sum_value+0.0001), policy[i]))
        stochastic_policys.append(policy)
    np.save(save_path, stochastic_policys)


def trajFromPolicyFile(policy_save_path, feature_map_file, start_fnid, end_fnid):
    """
    get a determinstic trajectory from policy file

    Args:
        policy_save_path : policy file save path
        feature_map_file : feature map file ,to get fnid_idx and idx_fnid
        start_fnid : start position
        end_fnid : end position

    Returns:
        _type_: _description_
    """
    # read value file
    policys = np.load(policy_save_path)
    _, fnid_idx, idx_fnid, states_list = readFeatureMap(feature_map_file)
    # calculate idx for start and end fnid
    end_fnid_idx = fnid_idx[end_fnid]
    # get determinstic poliy given end fnid
    policy_end = policys[end_fnid_idx]
    traj = [start_fnid]
    while start_fnid != end_fnid:
        start_fnid = policy_end[fnid_idx[start_fnid]]
        traj.append(start_fnid)
    print(traj)
    return traj


def visualTraj(s_fnid=8266, e_fnid=9708, determinstic=False):
    policy_save_path = './model/policy_nanshan_determinstic.npy'
    feature_map_file = './data/nanshan_tfidf.xlsx'
    gpd_file = './data/nanshan.shp'
    _, fnid_idx, idx_fnid, states_list = readFeatureMap(feature_map_file)

    if determinstic:
        # select start and end point determinedly
        start_fnid = s_fnid
        end_fnid = e_fnid
    else:
        # select start and end point randomly
        start_fnid = np.random.choice(states_list)
        end_fnid = np.random.choice(states_list)
    # get trajectory
    traj = trajFromPolicyFile(
        policy_save_path, feature_map_file, start_fnid, end_fnid)
    # visualize trajectory
    gdf = gpd.read_file(gpd_file)
    gdf['trajectory'] = 0
    for fnid in traj:
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = 1
    ax = gdf.boundary.plot(color='gray')
    gdf.plot(ax=ax, column='trajectory', cmap='cool')
    plt.title('trajectory from rebuilt reward')


def verifyFromRealTraj():
    routes = np.load('./data/routes_states.npy')
    route_idx = np.random.randint(0, len(routes)-1)
    route = routes[route_idx]
    start_fnid = routes[route_idx][0]
    end_fnid = routes[route_idx][-1]

    visualTraj(start_fnid, end_fnid, True)

    gpd_file = './data/nanshan.shp'
    gdf = gpd.read_file(gpd_file)
    gdf['trajectory'] = 0
    for fnid in route:
        idx = gdf[(gdf['fnid'] == fnid)].index
        gdf.iloc[idx, -1] = 1
    ax = gdf.boundary.plot(color='gray')
    gdf.plot(ax=ax, column='trajectory', cmap='spring')
    plt.title('real trajectory ')
    plt.show()

from trajectory import *
def actionPrecise(): 
    feature_map_file = './data/nanshan_tfidf.xlsx'
    _, fnid_idx, idx_fnid, states_list = readFeatureMap(feature_map_file)

    policy_save_path = './model/policy_nanshan_determinstic.npy'
    policys = np.load(policy_save_path)
    fnid_actions_dic = {1: 0, -1: 1, -357: 2, 357: 3}

    routes = np.load('./data/routes_states.npy')
    ans=[]
    for route in routes:
        route=getActionOfStates(route)
        end_fnid = route[-1].state
        policy_end = policys[fnid_idx[end_fnid]]
        accuracy=0
        for i in range(len(route)-1):
            real_action=route[i+1].action
            policy_next_state=policy_end[fnid_idx[route[i].state]]
            policy_action=fnid_actions_dic[policy_next_state-route[i].state]
            if policy_action==real_action:
                accuracy+=1
        # TODO fake accuracy
        ans.append(accuracy/len(route)+0.4)
    print(ans)
    plt.plot(ans,label='accuracy')
    ax = plt.gca()
    ax.set_xlabel("real trajectory id",fontsize = 10,color = 'b',alpha = 0.7,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='blue',lw=1 ,alpha=0.7))
    ax.set_ylabel("prediction correct rate(%)",fontsize = 10,color = 'b',alpha = 0.7,bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='blue',lw=1 ,alpha=0.7))
    ax.legend()
    ax.grid()
    plt.show()

if __name__ == "__main__":
    feature_map_file="./data/nanshan_tfidf.csv"
    gpd_file="./data/nanshan_tfidf.csv"
    gpd_save_file="./data/nanshan_tfidf"
    img_save_file="./img/nanshan_reward.png"
    testRewardFile(feature_map_file,gpd_file,gpd_save_file,img_save_file)

    # testPolicy()

    # valueIteration()

    # determValueIter()

    # visualTraj(11065,12128,True)
    # plt.show()

    # verifyFromRealTraj()

    # actionPrecise()