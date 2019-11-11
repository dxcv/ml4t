"""
Template for implementing QLearner  (c) 2015 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved
Template code for CS 4646/7646
Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.
We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.
-----do not edit anything above this line---
Student Name: Christopher Fleisher
GT User ID: cfleisher3
GT ID: 903421975
"""
import numpy as np
import random as rand


class QLearner(object):
    def __init__(self, num_states=100, num_actions=4, alpha=0.2,
                 gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        """
        Initializes Q table for given number of states and actions to zeros.
        Stores given params for QLearner instance.

        params:
        - num_states: number of states to consider (int)
        - num_actions: number of actions available (int)
        - alpha: learning rate [0, 1] (float)
        - gamma: discount rate [0, 1] (float)
        - rar: random action rate probability [0, 1] (float)
        - radr: random action decay rate after each update (float)
        - dyna: dyna updates for each regular update (int)
        - verbose: print debugging statements (bool)
        """
        self.Q = np.zeros((num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        tclr = 0.00001
        self.tclr = tclr
        self.Tc = np.full((num_states, num_actions, num_states), tclr)
        self.R = np.zeros((num_states, num_actions))
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

    def author(self):
        return 'cfleisher3'

    def querysetstate(self, s):
        self.s = s
        if rand.random() > self.rar:
            action = self.Q[s].argmax()
        else:
            action = rand.randint(0, self.num_actions-1)

        self.a = action

        if self.verbose:
            print(f's = {s}, a = {action}')

        return action

    def query(self, s_prime, r):
        if rand.random() > self.rar:
            action = self.Q[s_prime].argmax()
        else:
            action = rand.randint(0, self.num_actions-1)

        # Q[s,a] = (1-alpha)Q[s,a]+alpha(r+gammaQ[s_prime, a_prime])
        self.Q[self.s, self.a] = (1-self.alpha)*self.Q[self.s, self.a] + \
            self.alpha*(r+self.gamma*self.Q[s_prime, action])

        if self.dyna > 0:
            s = self.s
            a = self.a
            self.Tc[s, a, s_prime] += self.tclr
            t = self.Tc[s, a, s_prime]/self.Tc[s, a, :].sum()
            self.T[s, a, s_prime] = t
            self.R[s, a] = (1-self.alpha)*self.R[s, a] + self.alpha*r
            n = self.dyna
            while True:
                if n == 0:
                    break
                s_ridx = rand.randint(0, self.num_states-1)
                a_ridx = rand.randint(0, self.num_actions-1)
                s_pdyn = self.T[s_ridx, a_ridx, :].argmax()
                r_dyn = self.R[s_ridx, a_ridx]
                if rand.random() > self.rar:
                    a_pdyn = self.Q[s_pdyn].argmax()
                else:
                    a_pdyn = rand.randint(0, self.num_actions-1)

                self.Q[s_ridx, a_ridx] = (1-self.alpha)*self.Q[s_ridx, a_ridx]\
                    + self.alpha*(r_dyn+self.gamma*self.Q[s_pdyn, a_pdyn])
                n -= 1

        # reflect step
        self.s = s_prime
        self.a = action
        self.rar *= self.radr

        if self.verbose:
            print(f's = {s_prime}, a = {action}, r={r}')

        return action


if __name__ == "__main__":
    print('qlearner main...')
