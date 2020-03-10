# -*- coding:utf-8 -*-
from itertools import combinations
from itertools import permutations
from itertools import product
from collections import Counter
from copy import deepcopy
from functools import partial

FOLD = 0
CALL = 1
RAISE = 2

def overlap(t1, t2):
    for x in t1:
        if x in t2:
            return True
    return False

def all_unique(hc):
    for i in range(len(hc)-1):
        for j in range(i+1,len(hc)):
            if overlap(hc[i], hc[j]):
                return False
    return True

# 信息集表示
def default_infoset_format(player, holecards, board, bet_history):
    return "{0}{1}:{2}:".format("".join([str(x) for x in holecards]), "".join([str(x) for x in board]), bet_history)

# 定义游戏设置
class GameRules(object):
    def __init__(self, players, deck, rounds, ante, blinds, handeval, infoset_format):
        assert(players >= 2)  # player count
        assert(ante >= 0)
        assert(rounds != None)
        assert(deck != None)
        assert(len(rounds) > 0)
        assert(len(deck) > 1)
        if blinds != None:
            if type(blinds) is int or type(blinds) is float:
                blinds = [blinds]
        for r in rounds:
            assert(len(r.maxbets) == players)
        self.players = players
        self.deck = deck
        self.roundinfo = rounds
        self.ante = ante
        self.blinds = blinds
        self.handeval = handeval
        self.infoset_format = infoset_format

# 每回合的信息
class RoundInfo(object):
    def __init__(self, holecards, boardcards, betsize, maxbets):
        self.holecards = holecards
        self.boardcards = boardcards
        self.betsize = betsize
        self.maxbets = maxbets

# 博弈树
class GameTree(object):
    def __init__(self, rules):
        self.rules = deepcopy(rules)
        self.information_sets = {}
        self.root = None


