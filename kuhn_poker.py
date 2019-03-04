# -*- coding:utf-8 -*-
import random

PASS = 0
BET = 1
NUM_ACTIONS = 2

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.infoSet = ""
        self.regretSum = [0.0] * NUM_ACTIONS
        self.strategy = [0.0] * NUM_ACTIONS
        self.strategySum = [0.0] * NUM_ACTIONS

    def getStrategy(self,realizationWeight):
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]
        for a in range(NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1.0 / NUM_ACTIONS
            self.strategySum[a] += realizationWeight * self.strategy[a]
        return self.strategy

    def getAverageStrategy(self):
        avgStrategy = [0.0] * NUM_ACTIONS
        normalizingSum = 0
        for a in range(NUM_ACTIONS):
            normalizingSum += self.strategySum[a]
        for a in range(NUM_ACTIONS):
            if normalizingSum > 0:
                avgStrategy[a] = self.strategySum[a] / normalizingSum
            else:
                avgStrategy[a] = 1.0 / NUM_ACTIONS
        return avgStrategy

    def getInfo(self):
        return self.infoSet + ": " + str(self.getAverageStrategy())

def computer_terminal_state_payoff():
    pass

def cfr(cards, history, p0, p1):
    plays = len(history)

def train(iterations):
    cards = [1, 2, 3]
    util = 0.0
    for i in range(iterations):
        random.shuffle(cards)
        # for c1 in range(len(cards) - 1, -1, -1):
        #     c2 = random.randint(0, c1)
        #     temp = cards[c1]
        #     cards[c1] = cards[c2]
        #     cards[c2] = temp
        print(i + 1, cards)
        # util += cfr(cards, "", 1, 1)
    # print("Average game value: " + util / iterations)
    # Node traversal

if __name__ == "__main__":
    train(10)