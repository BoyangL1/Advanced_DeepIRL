import numpy as np


class RealGridWorld(object):
    def __init__(self, fnid_idx, idx_fnid, trans_prob):
        """
            modelling the real grid world
        Args:
            fnid_idx : {fnid:index} index start from 0
            trans_prob : probability that state tranform to another one when take action
        """
        self.fnid_idx = fnid_idx
        self.idx_fnid = idx_fnid
        self.trans_prob = trans_prob
        self.n_actions = 5
        self.actions = [0, 1, 2, 3, 4]
        self.neighbors = [1, -1, -357, 357, 0]
        self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 's'}
        #              right,    left,   down,   up ,   stay

    def get_transition_states_and_probs(self, state_fnid, action):
        """
            get all the possible transition states and their probabilities with [action] on [state]
        Args:
            state_fnid : fnid of current state
            action : int
        Return:
            a list of (state,probability) pair 
        """
        if self.trans_prob == 1:
            inc = self.neighbors[action]
            nei_s = state_fnid+inc
            if nei_s not in self.fnid_idx:
                return [(state_fnid, 1)]
            else:
                return [(nei_s, 1)]
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs[action] = self.trans_prob
            mov_probs += (1-self.trans_prob)/(self.n_actions-1)
            mov_probs[action] -= (1-self.trans_prob)/(self.n_actions-1)

            for a in range(len(self.actions)):
                inc = self.neighbors[a]
                nei_s = state_fnid+inc
                if nei_s not in self.fnid_idx:
                    mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0

            res = []
            for a in range(len(self.actions)):
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    nei_s = state_fnid+inc
                    res.append((nei_s, mov_probs[a]))
            return res

    def get_transition_mat(self):
        """
            get transition dynamics of the gridworld

        return:
            P_a         N_STATESDxN_STATESxN_ACTIONS transition probabilities matrix - 
                        P_a[s0, s1, a] is the transition prob of 
                        landing at state s1 when taking action 
                        a at state s0
        """
        N_STATES = len(self.fnid_idx)
        N_ACTIONS = self.n_actions
        P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))
        for si in range(N_STATES):
            posi = self.idx_fnid[si]
            for a in range(N_ACTIONS):
                probs = self.get_transition_states_and_probs(posi, a)
                for posj, prob in probs:
                    sj = self.fnid_idx[posj]
                    # Probility from si to sj given action a
                    P_a[si, sj, a] = prob
        return P_a
