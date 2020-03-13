# -*- coding:utf-8 -*-
from sample_CFR.pokerstrategy import *
from sample_CFR.pokergames import *
import random
import copy

class CounterfactualRegretMinimizer(object):
    def __init__(self, rules):
        self.rules = rules
        self.profile = StrategyProfile(rules, [Strategy(i) for i in range(rules.players)])
        self.current_profile = StrategyProfile(rules, [Strategy(i) for i in range(rules.players)])
        self.iterations = 0 # 迭代次数
        self.counterfactual_regret = [] # 反事实值
        self.action_reachprobs = [] # 到达概率
        self.tree = PublicTree(rules) # 建立公共树
        self.tree.build()
        print('Information sets: {0}'.format(len(self.tree.information_sets)))
        for s in self.profile.strategies: # 创建博弈树中决策节点的策略
            s.build_default(self.tree)
            self.counterfactual_regret.append({infoset: [0, 0, 0] for infoset in s.policy})
            self.action_reachprobs.append({infoset: [0, 0, 0] for infoset in s.policy})

    def run(self, num_iterations):
        for iteration in range(num_iterations):
            self.cfr()
            self.iterations += 1

    def cfr(self):
        self.cfr_helper(self.tree.root, [{(): 1} for _ in range(self.rules.players)]) # 根节点的到达概率为1

    def cfr_helper(self, root, reachprobs): # 递归方式更新, reach probs: 每位玩家的到达概率
        if type(root) is TerminalNode:
            return self.cfr_terminal_node(root, reachprobs)
        if type(root) is HolecardChanceNode:
            return self.cfr_holecard_node(root, reachprobs)
        if type(root) is BoardcardChanceNode:
            return self.cfr_boardcard_node(root, reachprobs)
        return self.cfr_action_node(root, reachprobs)

    def cfr_terminal_node(self, root, reachprobs):
        payoffs = [None for _ in range(self.rules.players)]
        for player in range(self.rules.players):
            player_payoffs = {hc: 0 for hc in root.holecards[player]}
            counts = {hc: 0 for hc in root.holecards[player]}
            for hands, winnings in root.payoffs.items():
                prob = 1.0
                player_hc = None
                for opp, hc in enumerate(hands):
                    if opp == player:
                        player_hc = hc
                    else:
                        prob *= reachprobs[opp][hc]
                player_payoffs[player_hc] += prob * winnings[player]
                counts[player_hc] += 1
            for hc, count in counts.items():
                if count > 0:
                    player_payoffs[hc] /= float(count)
            payoffs[player] = player_payoffs
        return payoffs

    def cfr_holecard_node(self, root, reachprobs):
        assert (len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - prevlen, root.todeal))
        next_reachprobs = [
            {hc: reachprobs[player][hc[0:prevlen]] / possible_deals for hc in root.children[0].holecards[player]} for
            player in range(self.rules.players)]
        subpayoffs = self.cfr_helper(root.children[0], next_reachprobs)
        payoffs = [{hc: 0 for hc in root.holecards[player]} for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand, winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def cfr_boardcard_node(self, root, reachprobs):
        prevlen = len(reachprobs[0].keys()[0])
        possible_deals = float(choose(len(root.deck) - prevlen, root.todeal))
        payoffs = [{hc: 0 for hc in root.holecards[player]} for player in range(self.rules.players)]
        for bc in root.children:
            next_reachprobs = [{hc: reachprobs[player][hc] / possible_deals for hc in bc.holecards[player]} for player
                               in range(self.rules.players)]
            subpayoffs = self.cfr_helper(bc, next_reachprobs)
            for player, subpayoff in enumerate(subpayoffs):
                for hand, winnings in subpayoff.items():
                    payoffs[player][hand] += winnings
        return payoffs

    def cfr_action_node(self, root, reachprobs):
        # Calculate strategy from counterfactual regret
        strategy = self.cfr_strategy_update(root, reachprobs) # 计算当前节点的策略, 类
        next_reachprobs = reachprobs
        action_probs = {hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for
                        hc in reachprobs[root.player]}
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = {hc: action_probs[hc][FOLD] * reachprobs[root.player][hc] for hc in
                                            reachprobs[root.player]}
            action_payoffs[FOLD] = self.cfr_helper(root.fold_action, next_reachprobs)
        if root.call_action:
            next_reachprobs[root.player] = {hc: action_probs[hc][CALL] * reachprobs[root.player][hc] for hc in
                                            reachprobs[root.player]}
            action_payoffs[CALL] = self.cfr_helper(root.call_action, next_reachprobs)
        if root.raise_action:
            next_reachprobs[root.player] = {hc: action_probs[hc][RAISE] * reachprobs[root.player][hc] for hc in
                                            reachprobs[root.player]}
            action_payoffs[RAISE] = self.cfr_helper(root.raise_action, next_reachprobs)
        payoffs = []
        for player in range(self.rules.players):
            player_payoffs = {hc: 0 for hc in reachprobs[player]}
            for i, subpayoff in enumerate(action_payoffs):
                if subpayoff is None:
                    continue
                for hc, winnings in subpayoff[player].items():
                    # action_probs is baked into reachprobs for everyone except the acting player
                    if player == root.player:
                        player_payoffs[hc] += winnings * action_probs[hc][i]
                    else:
                        player_payoffs[hc] += winnings
            payoffs.append(player_payoffs)
        # Update regret calculations
        self.cfr_regret_update(root, action_payoffs, payoffs[root.player])
        return payoffs

    def cfr_strategy_update(self, root, reachprobs):
        if self.iterations == 0:
            default_strat = self.profile.strategies[root.player]
            for hc in root.holecards[root.player]:
                infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
                probs = default_strat.probs(infoset)
                for i in range(3):
                    self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player][hc] * probs[i]
            return default_strat
        for hc in root.holecards[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            prev_cfr = self.counterfactual_regret[root.player][infoset]
            sumpos_cfr = sum([max(0, x) for x in prev_cfr])
            if sumpos_cfr == 0:
                probs = self.equal_probs(root)
            else:
                probs = [max(0, x) / sumpos_cfr for x in prev_cfr]
            self.current_profile.strategies[root.player].policy[infoset] = probs
            for i in range(3):
                self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player][hc] * probs[i]
            self.profile.strategies[root.player].policy[infoset] = [
                self.action_reachprobs[root.player][infoset][i] / sum(self.action_reachprobs[root.player][infoset]) for
                i in range(3)]
        return self.current_profile.strategies[root.player]

    def cfr_regret_update(self, root, action_payoffs, ev):
        for i, subpayoff in enumerate(action_payoffs):
            if subpayoff is None:
                continue
            for hc, winnings in subpayoff[root.player].items():
                immediate_cfr = winnings - ev[hc]
                infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
                self.counterfactual_regret[root.player][infoset][i] += immediate_cfr

    def equal_probs(self, root):
        total_actions = len(root.children)
        probs = [0, 0, 0]
        if root.fold_action:
            probs[FOLD] = 1.0 / total_actions
        if root.call_action:
            probs[CALL] = 1.0 / total_actions
        if root.raise_action:
            probs[RAISE] = 1.0 / total_actions
        return probs

def near(val, expected, distance=0.0001):
    return val >= (expected - distance) and val <= (expected + distance)

if __name__ == "__main__":
    print('')
    print('')
    print('Testing CFR')
    print('')
    print('')
    print('Computing NE for Half-Street Kuhn poker')

    hskuhn = half_street_kuhn_rules() # 游戏设置
    cfr = CounterfactualRegretMinimizer(hskuhn)
    iterations_per_block = 1000
    blocks = 10
    for block in range(blocks):
        print('Iterations: {0}'.format(block * iterations_per_block))
        cfr.run(iterations_per_block)
        result = cfr.profile.best_response()
        print('Best response EV: {0}'.format(result[1]))
        print('Total exploitability: {0}'.format(sum(result[1])))
    print(cfr.profile.strategies[0].policy)
    print(cfr.profile.strategies[1].policy)
    print(cfr.counterfactual_regret)
    print('Verifying P1 policy')
    assert (near(cfr.profile.strategies[0].policy['Q:/:'][CALL], 2.0 / 3.0, 0.01))
    assert (near(cfr.profile.strategies[0].policy['Q:/:'][RAISE], 1.0 / 3.0, 0.01))
    assert (near(cfr.profile.strategies[0].policy['K:/:'][CALL], 1, 0.01))
    assert (near(cfr.profile.strategies[0].policy['K:/:'][RAISE], 0, 0.01))
    assert (near(cfr.profile.strategies[0].policy['A:/:'][CALL], 0, 0.01))
    assert (near(cfr.profile.strategies[0].policy['A:/:'][RAISE], 1.0, 0.01))
    print('Verifying P2 policy')
    assert (near(cfr.profile.strategies[1].policy['Q:/r:'][FOLD], 1.0, 0.01))
    assert (near(cfr.profile.strategies[1].policy['Q:/r:'][CALL], 0, 0.01))
    assert (near(cfr.profile.strategies[1].policy['K:/r:'][FOLD], 2.0 / 3.0, 0.01))
    assert (near(cfr.profile.strategies[1].policy['K:/r:'][CALL], 1.0 / 3.0, 0.01))
    assert (near(cfr.profile.strategies[1].policy['A:/r:'][FOLD], 0, 0.01))
    assert (near(cfr.profile.strategies[1].policy['A:/r:'][CALL], 1.0, 0.01))

    print('Done!')
    print('')

    print('Computing NE for Leduc poker')
    leduc = leduc_rules()

    cfr = CounterfactualRegretMinimizer(leduc)

    iterations_per_block = 10
    blocks = 1000
    for block in range(blocks):
        print('Iterations: {0}'.format(block * iterations_per_block))
        cfr.run(iterations_per_block)
        result = cfr.profile.best_response()
        print('Best response EV: {0}'.format(result[1]))
        print('Total exploitability: {0}'.format(sum(result[1])))
    print('Done!')
    print('')




