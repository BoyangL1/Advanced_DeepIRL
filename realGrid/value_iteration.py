from email import policy
import math
import numpy as np
from collections import deque


def value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True):
    """
    static value iteration function. Perhaps the most useful function in this repo

    inputs:
      P_a         NxNxN_ACTIONS transition probabilities matrix - 
                                P_a[s0, s1, a] is the transition prob of 
                                landing at state s1 when taking action 
                                a at state s0
      rewards     Nx1 matrix - rewards for all the states
      gamma       float - RL discount
      error       float - threshold for a stop
      deterministic   bool - to return deterministic policy or stochastic policy

    returns:
      values    Nx1 matrix - estimated values
      policy    Nx1 (NxN_ACTIONS if non-det) matrix - policy
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    values = np.zeros([N_STATES])

    # estimate values
    while True:
        values_tmp = values.copy()

        for s in range(N_STATES):
            v_s = []
            values[s] = max([sum([P_a[s, s1, a]*(rewards[s] + gamma*values_tmp[s1])
                            for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])

        max_diff = np.max(np.abs(values - values_tmp))
        print(max_diff)
        if  max_diff < error:
            break


    if deterministic:
        # generate deterministic policy
        policy = np.zeros([N_STATES])
        for s in range(N_STATES):
            policy[s] = np.argmax([sum([P_a[s, s1, a]*(rewards[s]+gamma*values[s1])
                                        for s1 in range(N_STATES)])
                                   for a in range(N_ACTIONS)])
        return values, policy
    else:
        # generate stochastic policy
        policy = np.zeros([N_STATES, N_ACTIONS])
        for s in range(N_STATES):
            v_s = np.array([sum([P_a[s, s1, a]*(rewards[s] + gamma*values[s1])
                                 for s1 in range(N_STATES)])
                            for a in range(N_ACTIONS)])
            policy[s, :] = np.transpose(v_s/np.sum(v_s))
        return values, policy


def determinValIteration(end_fnid, actions, neighbors, gamma, fnid_idx, reward_map):
    """
    determinstic value iteration

    Args:
        end_fnid : destination fnid  
        actions : action list
        neighbors : neighboring fnid distance
        gamma : Attenuation coefficient .Default set to 0.9
        fnid_idx : fnid:idx
        reward_map : reward map

    Returns:
        values,policy: value map policy map
    """    
    values = [0]*len(fnid_idx)
    policy = [-1]*len(fnid_idx)
    # values[fnid_idx[end_fnid]] = float(reward_map[fnid_idx[end_fnid]])
    values[fnid_idx[end_fnid]] = reward_map[fnid_idx[end_fnid]]
    policy[fnid_idx[end_fnid]] = end_fnid
    queue = deque()
    queue.append(end_fnid)
    while queue:
        cur_fnid = queue.popleft()
        for a in actions:
            nei_fnid = cur_fnid+neighbors[a]
            if nei_fnid in fnid_idx.keys() and values[fnid_idx[nei_fnid]] == 0:
                queue.append(nei_fnid)
                nei_fnid_idx = fnid_idx[nei_fnid]

                max_value=float('-inf')
                max_fnid=0
                for a in actions:
                    n = nei_fnid+neighbors[a]
                    if n in fnid_idx.keys() and max_value<reward_map[nei_fnid_idx]+gamma*values[fnid_idx[n]]:
                        max_value=reward_map[nei_fnid_idx]+gamma*values[fnid_idx[n]] 
                        max_fnid=n
                values[nei_fnid_idx] = float(max_value)
                policy[nei_fnid_idx] = max_fnid
    return values,policy