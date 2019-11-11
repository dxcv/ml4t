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
        self.num_actions = num_actions
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
        """
        Used for setting the initial state and using a learned policy.

        Sets state returning either random or optimal policy action. Neither
        the Q table nor the random action rate are updated.

        params:
        - s: new state under consideration (int)

        returns:
        - a: random or optimal policy action for given state s
        """
        self.s = s
        if rand.random() > self.rar:
            action = self.Q[s].argmax()
        else:
            action = rand.randint(0, self.num_actions-1)

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):
        """
        Used for learning policy.

        Gets either optimal policy or random action. Updates Q table and tracks
        step by storing new state, action, and updating random action rate.

        params:
        - s_prime: new state (int)
        - r: reward for taking action at last state (float)

        returns:
        - a_prime: optimal policy action
        """
        if rand.random() > self.rar:
            action = self.Q[self.s].argmax()
        else:
            action = rand.randint(0, self.num_actions-1)

        # Q[s,a] = (1-alpha)Q[s,a]+alpha(r+gammaQ[s_prime, a_prime])
        a_prime = self.Q[s_prime].argmax()
        self.Q[self.s, action] = (1-self.alpha)*self.Q[self.s, action] + \
            self.alpha*(r+self.gamma*self.Q[s_prime, a_prime])

        # reflect step
        self.s = s_prime
        self.a = action
        self.rar *= self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {self.a}, r={r}")

        return action


if __name__ == "__main__":
    ql = QLearner(verbose=True)
    s = 99
    a = ql.querysetstate(s)
    s_prime = 5
    r = 0
    ql.query(s_prime, r)
