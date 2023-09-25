import imp
import numpy as np
from realGrid import real_grid
from realGrid import value_iteration
import img_utils
from utils import *
import time
from datetime import datetime


def compute_state_visition_freq(P_a, gamma, trajs, policy,fnid_idx, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)

    T = len(trajs[0])
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[fnid_idx[traj[0].state], 0] += 1
    mu[:, 0] = mu[:, 0]/len(trajs)

    for s in range(N_STATES):
        for t in range(T-1):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
                                 for pre_s in range(N_STATES)])
            else:
                mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1]
                                 for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def recursiveLogit(feat_map, genderAge, P_a, gamma, trajs, fnid_idx,lr, n_iters):
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    genderAge=np.array(genderAge)
    genderAge = np.tile(genderAge, (feat_map.shape[0], 1))
    feat_map = np.concatenate((genderAge, feat_map), axis=1)
    # init parameters
    theta = np.random.uniform(size=(feat_map.shape[1],))
    # calc feature expectations
    feat_exp = np.zeros([feat_map.shape[1]])

    for episode in trajs:
        for step in episode:
            feat_exp += feat_map[fnid_idx[step.state], :]
    feat_exp = feat_exp/len(trajs)
    
    # set pre-reward
    pre_reward=np.zeros(feat_map.shape[0])
    T0=time.time()
    now_time=datetime.now()
    print('this loop start at {}'.format(now_time))
    # training
    for iteration in range(n_iters):
        np.save('./model/rlogit_realworld.npy',theta)
        T1=time.time()
        print('iteration: {}/{}'.format(iteration, n_iters))
        print(iteration,n_iters)

        # compute reward function
        rewards = np.dot(feat_map, theta)
        reward_difference=np.mean(normalize(rewards)-pre_reward)
        print("the current reward difference is {}".format(reward_difference))
        if reward_difference<=0.001:
            np.save('./model/rlogit_realworld.npy',theta)
            print('the difference of reward is less than 0.001, then break the loop')
            break

        # compute policy
        _, policy = value_iteration.value_iteration(
            P_a, rewards, gamma, error=0.01, deterministic=False)

        # compute state visition frequences
        svf = compute_state_visition_freq(
            P_a, gamma, trajs, policy, fnid_idx,deterministic=False)

        # compute gradients
        grad = feat_exp - feat_map.T.dot(svf)

        # update params
        theta += lr * grad

        # calculate time pass
        T2=time.time()
        print("this iteration lasts {:.2f},the loop lasts {:.2f}".format(T2-T1,T2-T0))

        # set pre reward
        pre_reward=normalize(rewards)
    rewards = np.dot(feat_map, theta)
    return normalize(rewards)
