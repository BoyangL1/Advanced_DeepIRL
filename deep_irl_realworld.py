from turtle import st
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import realGrid.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *
import time
from traj_policy_logll import *

Step=namedtuple('Step',['state','action'])
class DeepIRLFC:
    def __init__(self, n_input, lr, genderAge, n_h1=400, n_h2=300, l2=0.5, name='deep_irl_fc'):
        """initialize DeepIRl, construct function between feature and reward

        Args:
            n_input (_type_): number of features
            lr : learning rate
            n_h1 (int, optional): output size of fc1. Defaults to 400.
            n_h2 (int, optional): output size of fc2. Defaults to 300.
            l2 (int, optional): l2 loss gradient. Defaults to 0.1.
            name (str, optional): variable scope. Defaults to 'deep_irl_fc'.
        """
        self.n_input = n_input
        self.lr = lr
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.name = name
        self.embedding_dim = 32
        self.genderAge = genderAge

        self.sess = tf.Session()
        self.input_s, self.reward, self.theta, self.input_onehot = self._build_network(
            self.name)
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        # apply l2 loss gradient
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(ys=self.l2_loss, xs=self.theta)
        self.grad_reward = tf.placeholder(tf.float32, [None, 1])
        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_reward)
        self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i])
                           for i in range(len(self.grad_l2))]
        # Gradient Clipping
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)
        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(
            zip(self.grad_theta, self.theta))

        self.sess.run(tf.global_variables_initializer())

    def _build_network(self, name):
        """build forward netword with 3 fully connected layer

        Args:
            name (string): variable scope

        Returns:
            input_s: features of states
            reward: reward of states
            theta: trainable parameters
            input_onehot= gender and age parameters
        """
        input_s = tf.placeholder(tf.float32, [None, self.n_input])
        input_onehot = tf.placeholder(tf.int32, [None, len(self.genderAge)])

        embedding_matrix = tf.get_variable("embedding_matrix", [self.embedding_dim])
        
        with tf.variable_scope(name):
            fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            
            embedded_input_onehot = tf.nn.embedding_lookup(embedding_matrix, input_onehot)
            fc3 = tf_utils.fc(embedded_input_onehot, self.n_h1, scope="fc3", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
            fc4 = tf_utils.fc(fc3, self.n_h2, scope="fc4", activation_fn=tf.nn.elu,
                              initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

            fc_contact = tf.concat(1, [fc2, fc4])

            fc_final = tf_utils.fc(fc_contact, self.n_h2, scope="fc_final", activation_fn=tf.nn.elu,
                                   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

            reward = tf_utils.fc(fc_final, 1, scope="reward")

        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return input_s, reward, theta, input_onehot

    def get_theta(self):
        return self.sess.run(self.theta)

    def get_rewards(self, states, oh):
        rewards = self.sess.run(self.reward, feed_dict={
                                self.input_s: states, self.input_onehot: oh})
        return rewards

    def apply_grads(self, feat_map, grad_r,oh):
        grad_r = np.reshape(grad_r, [-1, 1])
        feat_map = np.reshape(feat_map, [-1, self.n_input])
        _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
                                                           feed_dict={self.grad_reward: grad_r, self.input_s: feat_map, self.input_onehot: oh})
        return grad_theta, l2_loss, grad_norms

    def restoreGraph(self, model_file, model_name):
        saver = tf.train.Saver()
        save_path = model_file+".\\"+model_name
        saver.restore(self.sess, save_path)


def expectStateVisitFreq(P_a, gamma, trajs, fnid_idx, policy, deterministic=True):
    """
    compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of Steps - collected from expert
      fnid_idx {fnid:index}
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    T = []
    for traj in trajs:
        T.append(len(traj))

    avg_T = int(np.mean(T))
    # mu[s, t] is the prob of visiting state s at step t,get the
    mu = np.zeros([N_STATES, avg_T])

    for traj in trajs:
        index = fnid_idx[traj[0].state]
        mu[index, 0] += 1
    mu[:, 0] = mu[:, 0]/len(trajs)

    for s in range(N_STATES):
        for t in range(avg_T-1):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])]
                                 for pre_s in range(N_STATES)])
            else:
                mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)])
                                  for pre_s in range(N_STATES)])
    p = np.sum(mu, 1)
    return p


def stateVisitFreq(trajs, fnid_idx, n_states):
    """
    compute state visitation frequences from demonstrations

    input:
      trajs   list of list of Steps - collected from expert
      fnid_idx {fnid:index}
      n_states  number of states
    returns:
      p       Nx1 vector - state visitation frequences   
    """

    p = np.zeros(n_states)
    for traj in trajs:
        for step in traj:
            if step.state not in fnid_idx:
                continue
            idx = fnid_idx[step.state]
            p[idx] += 1
    p = p/len(trajs)
    return p


def deepMaxEntIRL(feat_map, P_a, gamma, trajs, lr, n_iters, fnid_idx, idx_fnid, gpd_file, genderAge, restore=True):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL), without personalized features

    inputs:
      feat_map    NxD matrix - the features for each state
      P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                         landing at state s1 when taking action 
                                         a at state s0
      gamma       float - RL discount factor
      trajs       a list of demonstrations
      lr          float - learning rate
      n_iters     int - number of optimization steps
      fnid_idx    {fnid:index}
      idx_fnid    {index:fnid}
      genderAge   one-hot code of gender&age e.g.[1,0,1,0,1]
    returns
      rewards     Nx1 vector - recoverred state rewards
    """

    # tf.set_random_seed(1)

    N_STATES, _, N_ACTIONS = np.shape(P_a)

    # init nn model
    nn_r = DeepIRLFC(feat_map.shape[1], lr, genderAge, 40, 30)

    # restor graph
    model_file = './model'
    model_name = 'realworld'
    if restore and os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)

    # find state visitation frequencies using demonstrations
    mu_D = stateVisitFreq(trajs, fnid_idx, N_STATES)

    # set pre-reward
    pre_reward = np.zeros(feat_map.shape[0])

    T0 = time.time()
    now_time = datetime.datetime.now()
    print('this loop start at {}'.format(now_time))
    # training
    for iteration in range(n_iters):
        T1 = time.time()
        if iteration % (n_iters/10) == 0:
            print('iteration: {}'.format(iteration))
            tf.train.Saver().save(nn_r.sess, './model/realworld')
        # compute the reward matrix
        rewards = nn_r.get_rewards(feat_map)
        reward_difference = np.mean(normalize(rewards)-pre_reward)
        print("the current reward difference is {}".format(reward_difference))
        if abs(reward_difference) <= 0.001:
            print('the difference of reward is less than 0.001, then break the loop')
            break
        # save picture
        img_utils.rewardVisual(normalize(rewards), idx_fnid,
                               gpd_file, "{} iteration".format(iteration))
        plt.savefig('./img/reward_{}.png'.format(iteration))
        plt.close()
        # compute policy
        values, policy = value_iteration.value_iteration(
            P_a, rewards, gamma, error=0.01, deterministic=True)
        np.save("./model/policy_realworld.npy", policy)
        print("The calculation of value and policy is finished!")
        # compute expected svf
        mu_exp = expectStateVisitFreq(
            P_a, gamma, trajs, fnid_idx, policy, deterministic=True)
        # compute gradients on rewards:
        grad_r = mu_D - mu_exp
        print("visit frequency difference is {}".format(np.mean(grad_r)))
        # apply gradients to the neural network
        _, _, _ = nn_r.apply_grads(feat_map, grad_r)
        # calculate time pass
        T2 = time.time()
        print("this iteration lasts {:.2f},the loop lasts {:.2f}".format(
            T2-T1, T2-T0))
        # set pre reward
        pre_reward = normalize(rewards)
    tf.train.Saver().save(nn_r.sess, './model/realworld')
    np.save('./model/policy_realworld.npy', policy)
    rewards = nn_r.get_rewards(feat_map)
    return normalize(rewards)


def deepMaxEntIRL2(nn_r,traj_file,feature_map_file, feat_map, P_a, gamma, trajs, lr, fnid_idx, idx_fnid, gpd_file, genderAge, restore):
    """
    Maximum Entropy Inverse Reinforcement Learning (Maxent IRL), with personalized features
    """    
    N_STATES, _, N_ACTIONS = np.shape(P_a)
    nn_r.genderAge = genderAge

    # restore graph
    model_file = './model'
    model_name = 'realworld'
    if restore and os.path.exists(os.path.join(model_file, model_name+'.meta')):
        print('restore graph from ckpt file')
        nn_r.restoreGraph(model_file, model_name)

    # find state visitation frequencies using demonstrations
    mu_D = stateVisitFreq(trajs, fnid_idx, N_STATES)
    # optimize the neural network by the difference between 
    # expected svf(state visit frequency) and real svf
    oh = [genderAge for _ in range(feat_map.shape[0])]
    rewards = normalize(nn_r.get_rewards(feat_map, oh))
    print("begin value iteration")
    values, policy = value_iteration.value_iteration(
        P_a, rewards, gamma, error=5, deterministic=True)
    np.save("./model/policy_realworld.npy", policy)
    print("The calculation of value and policy is finished!")
    # add traj log likelihood
    real_traj = trajFromExpert(traj_file)
    llrs = [trajLogLikelihood(feature_map_file, traj, trajFromPolicyFile(policy, fnid_idx, int(traj[0]), len(traj))) for traj in real_traj]
    print("The log-likelihood ratio of the generated trajectory to the true trajectory is {}".format(np.nanmean(llrs)))
    # compute expected svf
    mu_exp = expectStateVisitFreq(
        P_a, gamma, trajs, fnid_idx, policy, deterministic=True)
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp
    print("visit frequency difference is {}".format(np.mean(grad_r)))
    # apply gradients to the neural network
    _,_,_ = nn_r.apply_grads(feat_map, grad_r, oh)

    # Store model weights 
    tf.train.Saver().save(nn_r.sess, './model/realworld') # ckpt
    np.save('./model/policy_realworld.npy', policy) # npy

    return normalize(rewards), nn_r