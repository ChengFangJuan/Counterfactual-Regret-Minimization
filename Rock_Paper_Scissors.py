# -*- coding:utf-8 -*-
from __future__ import division
import random


Rock = 0
Paper = 1
Scissors = 2
Num_action = 3


class RockPaperScissor():

    def __init__(self):
        self.regretSum = [0] * Num_action
        self.strategy = [0] * Num_action
        self.strategySum = [0] * Num_action
        self.opp_strategy = [0.4, 0.3, 0.3]

    def get_strategy(self):
        normalizingSum = 0
        for a in range(Num_action):
            if self.regretSum[a] > 0:
                self.strategy[a] = self.regretSum[a]
            else:
                self.strategy[a] = 0
            normalizingSum += self.strategy[a]

        for a in range(Num_action):
            if normalizingSum > 0:
                self.strategy[a] = self.strategy[a] / normalizingSum
            else:
                self.strategy[a] = 1 / Num_action
            self.strategySum[a] += self.strategy[a]
        return self.strategy

    def get_action(self, strategy):
        r = random.random()
        action = 0
        cumulativerProbability = 0
        while action < Num_action -1:
            cumulativerProbability += strategy[action]
            if r < cumulativerProbability:
                break
            action += 1
        return action

    


