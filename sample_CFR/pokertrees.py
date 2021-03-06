# -*- coding:utf-8 -*-
from itertools import combinations
from itertools import permutations
from itertools import product
from collections import Counter
from functools import partial
import copy

FOLD = 0
CALL = 1
RAISE = 2


def overlap(t1, t2):
    for x in t1:
        if x in t2:
            return True
    return False


def all_unique(hc):
    for i in range(len(hc) - 1):
        for j in range(i + 1, len(hc)):
            if overlap(hc[i], hc[j]):
                return False
    return True


# 信息集表示
def default_infoset_format(player, holecards, board, bet_history):
    return "{0}{1}:{2}:".format("".join([str(x) for x in holecards]), "".join([str(x) for x in board]), bet_history)


# 定义游戏设置
class GameRules(object):
    def __init__(self, players, deck, rounds, ante, blinds, handeval=None, infoset_format=default_infoset_format):
        '''player: 玩家数
           deck: 游戏中所有的牌，list，元素为类
           rounds: 每个回合的信息，list，元素为类
           ante: 最小加注数
           blinds: 大小盲注，list
           handeval: 评估函数
           infoset_format: 信息集表示方法'''
        assert (players >= 2)  # player count
        assert (ante >= 0)
        assert (rounds != None)
        assert (deck != None)
        assert (len(rounds) > 0)
        assert (len(deck) > 1)
        if blinds != None:
            if type(blinds) is int or type(blinds) is float:
                blinds = [blinds]
        for r in rounds:
            assert (len(r.maxbets) == players)
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
        self.holecards = holecards  # 本回合手牌数
        self.boardcards = boardcards  # 本回合公共牌数
        self.betsize = betsize
        self.maxbets = maxbets  # 本回合每个玩家的可加注到的最大筹码


# 博弈树
class GameTree(object):
    # rules 为定义的游戏设置
    def __init__(self, rules):
        # rules type: 类, 定义游戏的规则
        self.rules = rules
        self.information_sets = {}
        self.root = None

    def build(self):
        # Assume everyone is in
        players_in = [True] * self.rules.players
        # Collect antes
        committed = [self.rules.ante] * self.rules.players
        bets = [0] * self.rules.players
        # Collect blinds
        next_player = self.collect_blinds(committed, bets, 0)
        holes = [()] * self.rules.players
        board = ()
        bet_history = ""
        self.root = self.build_rounds(None, players_in, committed, holes, board, self.rules.deck, bet_history, 0, bets,
                                      next_player)

    def collect_blinds(self, committed, bets, next_player):
        if self.rules.blinds != None:
            for blind in self.rules.blinds:
                committed[next_player] += blind
                bets[next_player] = int((committed[next_player] - self.rules.ante) / self.rules.roundinfo[0].betsize)
                next_player = (next_player + 1) % self.rules.players
        return next_player

    def deal_holecards(self, deck, holecards, players):
        a = combinations(deck, holecards)
        return filter(lambda x: all_unique(x), permutations(a, players))

    def build_rounds(self, root, players_in, committed, holes, board, deck, bet_history, round_idx, bets=None,
                     next_player=0):
        # root:根节点(初始None),players_in:玩家是否在玩,committed:累计加注筹码,holes:手牌,board:公共牌,deck:所有牌,bet_history:历史动作,round_idx:本回合编号,bets:加注筹码,next_player:玩家编号
        if round_idx == len(self.rules.roundinfo): # 表示游戏结束，创建终结点
            return self.showdown(root, players_in, committed, holes, board, deck, bet_history)
        bet_history += "/"
        cur_round = self.rules.roundinfo[round_idx]
        while not players_in[next_player]: # 如果此玩家弃牌，则下移到下一位玩家做决策
            next_player = (next_player + 1) % self.rules.players
        if bets is None:
            bets = [0] * self.rules.players
        min_actions_this_round = players_in.count(True)  # 此回合的最少动作数，即目前还存在多少玩家
        actions_this_round = 0
        if cur_round.holecards:  # 如果此回合存在手牌，则先建立发手牌机会节点
            return self.build_holecards(root, next_player, players_in, committed, holes, board, deck, bet_history,
                                        round_idx, min_actions_this_round, actions_this_round, bets)
        if cur_round.boardcards:
            return self.build_boardcards(root, next_player, players_in, committed, holes, board, deck, bet_history,
                                         round_idx, min_actions_this_round, actions_this_round, bets)
        return self.build_bets(root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                               min_actions_this_round, actions_this_round, bets)

    def get_next_player(self, cur_player, players_in):
        next_player = (cur_player + 1) % self.rules.players
        while not players_in[next_player]:
            next_player = (next_player + 1) % self.rules.players
        return next_player

    def build_holecards(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]
        hnode = HolecardChanceNode(root, committed, holes, board, self.rules.deck, "", cur_round.holecards)
        # Deal holecards
        all_hc = self.deal_holecards(deck, cur_round.holecards, players_in.count(True))
        # Create a child node for every possible distribution
        for cur_holes in all_hc:
            dealt_cards = ()
            cur_holes = list(cur_holes)
            cur_idx = 0
            for i, hc in enumerate(holes):
                # Only deal cards to players who are still in
                if players_in[i]:
                    cur_holes[cur_idx] = hc + cur_holes[cur_idx]
                    cur_idx += 1
            for hc in cur_holes:
                dealt_cards += hc
            cur_deck = filter(lambda x: not (x in dealt_cards), deck)
            if cur_round.boardcards:
                self.build_boardcards(hnode, next_player, players_in, committed, cur_holes, board, cur_deck,
                                      bet_history, round_idx, min_actions_this_round, actions_this_round, bets)
            else:
                self.build_bets(hnode, next_player, players_in, committed, cur_holes, board, cur_deck, bet_history,
                                round_idx, min_actions_this_round, actions_this_round, bets)
        return hnode

    def build_boardcards(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                         min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]
        bnode = BoardcardChanceNode(root, committed, holes, board, deck, bet_history, cur_round.boardcards)
        all_bc = combinations(deck, cur_round.boardcards)
        for bc in all_bc:
            cur_board = board + bc
            cur_deck = filter(lambda x: not (x in bc), deck)
            self.build_bets(bnode, next_player, players_in, committed, holes, cur_board, cur_deck, bet_history,
                            round_idx, min_actions_this_round, actions_this_round, bets)
        return bnode

    def build_bets(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                   min_actions_this_round, actions_this_round, bets_this_round): # holes: 双方玩家所有可能手牌
        # if everyone else folded, end the hand
        if players_in.count(True) == 1: # 如果玩家弃牌，游戏结束，创建终结点
            self.showdown(root, players_in, committed, holes, board, deck, bet_history)
            return
        # if everyone checked or the last raisor has been called, end the round # 结束本回合，进入下一回合
        if actions_this_round >= min_actions_this_round and self.all_called_last_raisor_or_folded(players_in,
                                                                                                  bets_this_round):
            self.build_rounds(root, players_in, committed, holes, board, deck, bet_history, round_idx + 1)
            return
        cur_round = self.rules.roundinfo[round_idx]
        anode = ActionNode(root, committed, holes, board, deck, bet_history, next_player, self.rules.infoset_format)
        # add the node to the information set
        if not (anode.player_view in self.information_sets):
            self.information_sets[anode.player_view] = []
        self.information_sets[anode.player_view].append(anode)
        # get the next player to act
        next_player = self.get_next_player(next_player, players_in)
        # add a folding option if someone has bet more than this player
        if committed[anode.player] < max(committed): # 双方累积加注筹码相等时，弃牌动作不合法
            self.add_fold_child(anode, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                                min_actions_this_round, actions_this_round, bets_this_round)
        # add a calling/checking option
        self.add_call_child(anode, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                            min_actions_this_round, actions_this_round, bets_this_round)
        # add a raising option if this player has not reached their max bet level
        if cur_round.maxbets[anode.player] > max(bets_this_round):
            self.add_raise_child(anode, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                                 min_actions_this_round, actions_this_round, bets_this_round)
        return anode

    def all_called_last_raisor_or_folded(self, players_in, bets):
        betlevel = max(bets)
        for i, v in enumerate(bets):
            if players_in[i] and bets[i] < betlevel:
                return False
        return True

    def add_fold_child(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                       min_actions_this_round, actions_this_round, bets_this_round):
        players_in[root.player] = False
        bet_history += 'f'
        self.build_bets(root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round + 1, bets_this_round)
        root.fold_action = root.children[-1]
        players_in[root.player] = True

    def add_call_child(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                       min_actions_this_round, actions_this_round, bets_this_round):
        player_commit = committed[root.player]
        player_bets = bets_this_round[root.player]
        committed[root.player] = max(committed)
        bets_this_round[root.player] = max(bets_this_round)
        bet_history += 'c'
        self.build_bets(root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round + 1, bets_this_round)
        root.call_action = root.children[-1]
        committed[root.player] = player_commit
        bets_this_round[root.player] = player_bets

    def add_raise_child(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round, bets_this_round):
        cur_round = self.rules.roundinfo[round_idx]
        prev_betlevel = bets_this_round[root.player]
        prev_commit = committed[root.player]
        bets_this_round[root.player] = max(bets_this_round) + 1
        committed[root.player] += (bets_this_round[root.player] - prev_betlevel) * cur_round.betsize
        bet_history += 'r'
        self.build_bets(root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round + 1, bets_this_round)
        root.raise_action = root.children[-1]
        bets_this_round[root.player] = prev_betlevel
        committed[root.player] = prev_commit

    def showdown(self, root, players_in, committed, holes, board, deck, bet_history):
        if players_in.count(True) == 1:
            winners = [i for i, v in enumerate(players_in) if v]
        else:
            scores = [self.rules.handeval(hc, board) for hc in holes]
            winners = []
            maxscore = -1
            for i, s in enumerate(scores):
                if players_in[i]:
                    if len(winners) == 0 or s > maxscore:
                        maxscore = s
                        winners = [i]
                    elif s == maxscore:
                        winners.append(i)
        pot = sum(committed)
        payoff = pot / float(len(winners))
        payoffs = [-x for x in committed]
        for w in winners:
            payoffs[w] += payoff
        return TerminalNode(root, committed, holes, board, deck, bet_history, payoffs, players_in)

    def holecard_distributions(self):
        x = Counter(combinations(self.rules.deck, self.holecards))
        d = float(sum(x.values()))
        return zip(x.keys(), [y / d for y in x.values()])


def multi_infoset_format(base_infoset_format, player, holecards, board, bet_history):
    return tuple([base_infoset_format(player, hc, board, bet_history) for hc in holecards])


class PublicTree(GameTree):
    def __init__(self, rules):
        GameTree.__init__(self, GameRules(rules.players, rules.deck, rules.roundinfo, rules.ante, rules.blinds,
                                          rules.handeval, partial(multi_infoset_format, rules.infoset_format)))

    def build(self):
        # Assume everyone is in
        players_in = [True] * self.rules.players
        # Collect antes
        committed = [self.rules.ante] * self.rules.players  # 累计的加注筹码
        bets = [0] * self.rules.players
        # Collect blinds
        next_player = self.collect_blinds(committed, bets, 0)
        holes = [[()]] * self.rules.players
        board = ()
        bet_history = ""
        self.root = self.build_rounds(None, players_in, committed, holes, board, self.rules.deck, bet_history, 0, bets,
                                      next_player)

    def build_holecards(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                        min_actions_this_round, actions_this_round, bets):
        # min_actions_this_round:此回合最少动作数, actions_this_round:本回合执行动作数
        cur_round = self.rules.roundinfo[round_idx]
        hnode = HolecardChanceNode(root, committed, holes, board, self.rules.deck, "", cur_round.holecards) # 创建手牌机会节点
        # Deal holecards
        all_hc = list(combinations(deck, cur_round.holecards))  #所有可能手牌的组合
        updated_holes = []
        for player in range(self.rules.players):
            if not players_in[player]:
                # Only deal to players who are still in the hand
                updated_holes.append([old_hc for old_hc in holes[player]])
            elif len(holes[player]) == 0:
                # If this player has no cards, just set their holecards to be the newly dealt ones
                updated_holes.append([new_hc for new_hc in all_hc])
            else:
                updated_holes.append([])
                # Filter holecards to valid combinations
                # TODO: Speed this up by removing duplicate holecard combinations
                for new_hc in all_hc:
                    for old_hc in holes[player]:
                        if not overlap(old_hc, new_hc):
                            updated_holes[player].append(old_hc + new_hc)
        if cur_round.boardcards:
            self.build_boardcards(hnode, next_player, players_in, committed, updated_holes, board, deck, bet_history,
                                  round_idx, min_actions_this_round, actions_this_round, bets)
        else:
            self.build_bets(hnode, next_player, players_in, committed, updated_holes, board, deck, bet_history,
                            round_idx, min_actions_this_round, actions_this_round, bets)
        return hnode

    def build_boardcards(self, root, next_player, players_in, committed, holes, board, deck, bet_history, round_idx,
                         min_actions_this_round, actions_this_round, bets):
        cur_round = self.rules.roundinfo[round_idx]
        bnode = BoardcardChanceNode(root, committed, holes, board, deck, bet_history, cur_round.boardcards)
        all_bc = combinations(deck, cur_round.boardcards)
        for bc in all_bc:
            cur_board = board + bc
            cur_deck = filter(lambda x: not (x in bc), deck)
            updated_holes = []
            # Filter any holecards that are now impossible
            for player in range(self.rules.players):
                updated_holes.append([])
                for hc in holes[player]:
                    if not overlap(hc, bc):
                        updated_holes[player].append(hc)
            self.build_bets(bnode, next_player, players_in, committed, updated_holes, cur_board, cur_deck, bet_history,
                            round_idx, min_actions_this_round, actions_this_round, bets)
        return bnode

    def showdown(self, root, players_in, committed, holes, board, deck, bet_history):
        # TODO: Speedup
        # - Pre-order list of hands
        pot = sum(committed)
        showdowns_possible = self.showdown_combinations(holes)
        if players_in.count(True) == 1:
            fold_payoffs = [-x for x in committed]
            fold_payoffs[players_in.index(True)] += pot
            payoffs = {hands: fold_payoffs for hands in showdowns_possible}
        else:
            scores = {}
            for i in range(self.rules.players):
                if players_in[i]:
                    for hc in holes[i]:
                        if not (hc in scores):
                            scores[hc] = self.rules.handeval(hc, board)
            payoffs = {hands: self.calc_payoffs(hands, scores, players_in, committed, pot) for hands in
                       showdowns_possible}
        return TerminalNode(root, committed, holes, board, deck, bet_history, payoffs, players_in)

    def showdown_combinations(self, holes):
        # Get all the possible holecard matchups for a given showdown.
        # Every card must be unique because two players cannot have the same holecard.
        return list(filter(lambda x: all_unique(x), product(*holes)))

    def calc_payoffs(self, hands, scores, players_in, committed, pot):
        winners = []
        maxscore = -1
        for i, hand in enumerate(hands):
            if players_in[i]:
                s = scores[hand]
                if len(winners) == 0 or s > maxscore:
                    maxscore = s
                    winners = [i]
                elif s == maxscore:
                    winners.append(i)
        payoff = pot / float(len(winners))
        payoffs = [-x for x in committed]
        for w in winners:
            payoffs[w] += payoff
        return payoffs


class Node(object):
    def __init__(self, parent, committed, holecards, board, deck, bet_history):
        self.committed = copy.deepcopy(committed)
        self.holecards = holecards
        self.board = board
        self.deck = deck
        self.bet_history = copy.deepcopy(bet_history)
        if parent: # 如果父节点存在，此节点加入到父节点的子节点列表中
            self.parent = parent
            self.parent.add_child(self)

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)


class TerminalNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, payoffs, players_in):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.payoffs = payoffs
        self.players_in = copy.deepcopy(players_in)


class HolecardChanceNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, todeal):
        # todeal:发几张手牌
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.todeal = todeal
        self.children = []


class BoardcardChanceNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, todeal):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.todeal = todeal
        self.children = []


class ActionNode(Node):
    def __init__(self, parent, committed, holecards, board, deck, bet_history, player, infoset_format):
        Node.__init__(self, parent, committed, holecards, board, deck, bet_history)
        self.player = player
        self.children = []
        self.raise_action = None
        self.call_action = None
        self.fold_action = None
        self.player_view = infoset_format(player, holecards[player], board, bet_history)

    def valid(self, action):
        if action == FOLD:
            return self.fold_action
        if action == CALL:
            return self.call_action
        if action == RAISE:
            return self.raise_action
        raise Exception("Unknown action {0}. Action must be FOLD, CALL, or RAISE".format(action))

    def get_child(self, action):
        return self.valid(action)
