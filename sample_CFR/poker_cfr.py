# -*- coding:utf-8 -*-
from sample_CFR.pokerstrategy import *
from sample_CFR.pokergames import *
import random
import copy


class CounterfactualRegretMinimizer(object):
    def __init__(self, rules):
        self.rules = rules  # 游戏规则设置
        self.profile = StrategyProfile(rules, [Strategy(i) for i in range(rules.players)])
        self.current_profile = StrategyProfile(rules, [Strategy(i) for i in range(rules.players)])
        self.iterations = 0  # 迭代次数
        self.counterfactual_regret = []  # 反事实值
        self.action_reachprobs = []  # 到达概率
        self.tree = PublicTree(rules)  # 建立公共树
        self.tree.build()
        print('Information sets: {0}'.format(len(self.tree.information_sets)))
        for s in self.profile.strategies:  # 创建博弈树中决策节点的策略
            s.build_default(self.tree)
            self.counterfactual_regret.append({infoset: [0, 0, 0] for infoset in s.policy})
            self.action_reachprobs.append({infoset: [0, 0, 0] for infoset in s.policy})

    def run(self, num_iterations):
        for iteration in range(num_iterations):
            self.cfr()
            self.iterations += 1

    def cfr(self):
        self.cfr_helper(self.tree.root, [{(): 1} for _ in range(self.rules.players)])  # 根节点的到达概率为1

    def cfr_helper(self, root, reachprobs):  # 递归方式更新, reach probs: 每位玩家的到达概率，列表
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
            player_payoffs = {hc: 0 for hc in root.holecards[player]}  # 玩家每种手牌下的收益
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
                    player_payoffs[hc] /= float(count)  # 计算平均收益
            payoffs[player] = player_payoffs
        return payoffs

    def cfr_holecard_node(self, root, reachprobs):  # 初始发手牌的节点
        assert (len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])  # 之前已经发出牌的个数
        possible_deals = float(choose(len(root.deck) - prevlen, root.todeal))  # 目前所有可能牌的组合数
        next_reachprobs = [
            {hc: reachprobs[player][hc[0:prevlen]] / possible_deals for hc in root.children[0].holecards[player]} for
            player in range(self.rules.players)]  # 计算到达下一时刻的概率，发牌动作服从均匀分布
        subpayoffs = self.cfr_helper(root.children[0], next_reachprobs)
        payoffs = [{hc: 0 for hc in root.holecards[player]} for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand, winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def cfr_boardcard_node(self, root, reachprobs):
        prevlen = len(list(reachprobs[0].keys())[0])
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
        strategy = self.cfr_strategy_update(root, reachprobs)  # 计算当前节点的策略, 类
        next_reachprobs = reachprobs
        action_probs = {hc: strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history)) for
                        hc in reachprobs[root.player]}
        action_payoffs = [None, None, None]  # 表示每个动作的收益
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

    def cfr_strategy_update(self, root, reachprobs):  # 策略更新
        if self.iterations == 0:  # 迭代次数为0，获得初始化的策略
            default_strat = self.profile.strategies[root.player]
            for hc in root.holecards[root.player]:
                infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
                probs = default_strat.probs(infoset)
                for i in range(3):
                    self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player][hc] * probs[i]
            return default_strat  # default_strat 初始化的策略
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


class PublicChanceSamplingCFR(CounterfactualRegretMinimizer):
    def __init__(self, rules):
        CounterfactualRegretMinimizer.__init__(self, rules)

    def cfr(self):
        # Sample all board cards to be used
        self.board = random.sample(self.rules.deck, sum([x.boardcards for x in self.rules.roundinfo]))
        # Call the standard CFR algorithm
        self.cfr_helper(self.tree.root, [{(): 1} for _ in range(self.rules.players)])

    def cfr_terminal_node(self, root, reachprobs):
        payoffs = [None for _ in range(self.rules.players)]
        for player in range(self.rules.players):
            player_payoffs = {hc: 0 for hc in reachprobs[player]}
            counts = {hc: 0 for hc in reachprobs[player]}
            for hands, winnings in root.payoffs.items():
                if not self.terminal_match(hands):
                    continue
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

    def terminal_match(self, hands):
        for hc in hands:
            if self.has_boardcard(hc):
                return False
        return True

    def cfr_holecard_node(self, root, reachprobs):
        assert (len(root.children) == 1)
        prevlen = len(list(reachprobs[0].keys())[0])
        possible_deals = float(choose(len(root.deck) - len(self.board) - prevlen, root.todeal))
        next_reachprobs = [
            {hc: reachprobs[player][hc[0:prevlen]] / possible_deals for hc in root.children[0].holecards[player] if
             not self.has_boardcard(hc)} for player in range(self.rules.players)]
        subpayoffs = self.cfr_helper(root.children[0], next_reachprobs)
        payoffs = [{hc: 0 for hc in reachprobs[player]} for player in range(self.rules.players)]
        for player, subpayoff in enumerate(subpayoffs):
            for hand, winnings in subpayoff.items():
                hc = hand[0:prevlen]
                payoffs[player][hc] += winnings
        return payoffs

    def has_boardcard(self, hc):
        for c in hc:
            if c in self.board:
                return True
        return False

    def cfr_boardcard_node(self, root, reachprobs):
        # Number of community cards dealt this round
        num_dealt = len(root.children[0].board) - len(root.board)
        # Find the child that matches the sampled board card(s)
        for bc in root.children:
            if self.boardmatch(num_dealt, bc):
                # Update the probabilities for each HC. Assume chance prob = 1 and renormalize reach probs by new holecard range
                # next_reachprobs = [{ hc: reachprobs[player][hc] for hc in reachprobs[player] } for player in range(self.rules.players)]
                # sumprobs = [sum(next_reachprobs[player].values()) for player in range(self.rules.players)]
                # if min(sumprobs) == 0:
                #    return [{ hc: 0 for hc in reachprobs[player] } for player in range(self.rules.players)]
                # next_reachprobs = [{ hc: reachprobs[player][hc] / sumprobs[player] for hc in bc.holecards[player] } for player in range(self.rules.players)]
                # Perform normal CFR
                results = self.cfr_helper(bc, reachprobs)
                # Return the payoffs
                return results
        raise Exception('Sampling from impossible board card')

    def boardmatch(self, num_dealt, node):
        # Checks if this node is a match for the sampled board card(s)
        for next_card in range(0, len(node.board)):
            if self.board[next_card] not in node.board:
                return False
        return True

    def cfr_strategy_update(self, root, reachprobs):
        # Update the strategies and regrets for each infoset
        for hc in reachprobs[root.player]:
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            # Get the current CFR
            prev_cfr = self.counterfactual_regret[root.player][infoset]
            # Get the total positive CFR
            sumpos_cfr = float(sum([max(0, x) for x in prev_cfr]))
            if sumpos_cfr == 0:
                # Default strategy is equal probability
                probs = self.equal_probs(root)
            else:
                # Use the strategy that's proportional to accumulated positive CFR
                probs = [max(0, x) / sumpos_cfr for x in prev_cfr]
            # Use the updated strategy as our current strategy
            self.current_profile.strategies[root.player].policy[infoset] = probs
            # Update the weighted policy probabilities (used to recover the average strategy)
            for i in range(3):
                self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player][hc] * probs[i]
            if sum(self.action_reachprobs[root.player][infoset]) == 0:
                # Default strategy is equal weight
                self.profile.strategies[root.player].policy[infoset] = self.equal_probs(root)
            else:
                # Recover the weighted average strategy
                self.profile.strategies[root.player].policy[infoset] = [
                    self.action_reachprobs[root.player][infoset][i] / sum(self.action_reachprobs[root.player][infoset])
                    for i in range(3)]
        # Return and use the current CFR strategy
        return self.current_profile.strategies[root.player]


class ChanceSamplingCFR(CounterfactualRegretMinimizer):
    def __init__(self, rules):
        CounterfactualRegretMinimizer.__init__(self, rules)

    def cfr(self):
        # Sample all cards to be used
        holecards_per_player = sum([x.holecards for x in self.rules.roundinfo])  # 整局游戏中手牌数目
        boardcards_per_hand = sum([x.boardcards for x in self.rules.roundinfo])  # 整局游戏中公共牌的数目
        todeal = random.sample(self.rules.deck, boardcards_per_hand + holecards_per_player * self.rules.players)
        # Deal holecards
        self.holecards = [tuple(todeal[p * holecards_per_player:(p + 1) * holecards_per_player]) for p in
                          range(self.rules.players)]
        self.board = tuple(todeal[-boardcards_per_hand:])
        # Set the top card of the deck
        self.top_card = len(todeal) - boardcards_per_hand
        # Call the standard CFR algorithm
        self.cfr_helper(self.tree.root, [1 for _ in range(self.rules.players)])

    def cfr_terminal_node(self, root, reachprobs):
        payoffs = [0 for _ in range(self.rules.players)]
        for hands, winnings in root.payoffs.items():
            if not self.terminal_match(hands):
                continue
            for player in range(self.rules.players):
                prob = 1.0
                for opp, hc in enumerate(hands):
                    if opp != player:
                        prob *= reachprobs[opp]
                payoffs[player] = prob * winnings[player]
            return payoffs

    def terminal_match(self, hands):
        for p in range(self.rules.players):
            if not self.hcmatch(hands[p], p):
                return False
        return True

    def hcmatch(self, hc, player):
        # Checks if this hand is isomorphic to the sampled hand
        sampled = self.holecards[player][:len(hc)]
        for c in hc:
            if c not in sampled:
                return False
        return True

    def cfr_holecard_node(self, root, reachprobs):
        assert (len(root.children) == 1)
        return self.cfr_helper(root.children[0], reachprobs)

    def cfr_boardcard_node(self, root, reachprobs):
        # Number of community cards dealt this round
        num_dealt = len(root.children[0].board) - len(root.board)
        # Find the child that matches the sampled board card(s)
        for bc in root.children:
            if self.boardmatch(num_dealt, bc):
                # Perform normal CFR
                results = self.cfr_helper(bc, reachprobs)
                # Return the payoffs
                return results
        raise Exception('Sampling from impossible board card')

    def boardmatch(self, num_dealt, node):
        # Checks if this node is a match for the sampled board card(s)
        for next_card in range(0, len(node.board)):
            if self.board[next_card] not in node.board:
                return False
        return True

    def cfr_action_node(self, root, reachprobs):
        # Calculate strategy from counterfactual regret
        strategy = self.cfr_strategy_update(root, reachprobs)
        next_reachprobs = reachprobs
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        action_probs = strategy.probs(self.rules.infoset_format(root.player, hc, root.board, root.bet_history))
        action_payoffs = [None, None, None]
        if root.fold_action:
            next_reachprobs[root.player] = action_probs[FOLD] * reachprobs[root.player]
            action_payoffs[FOLD] = self.cfr_helper(root.fold_action, next_reachprobs)
        if root.call_action:
            next_reachprobs[root.player] = action_probs[CALL] * reachprobs[root.player]
            action_payoffs[CALL] = self.cfr_helper(root.call_action, next_reachprobs)
        if root.raise_action:
            next_reachprobs[root.player] = action_probs[RAISE] * reachprobs[root.player]
            action_payoffs[RAISE] = self.cfr_helper(root.raise_action, next_reachprobs)
        payoffs = [0 for player in range(self.rules.players)]
        for i, subpayoff in enumerate(action_payoffs):
            if subpayoff is None:
                continue
            for player, winnings in enumerate(subpayoff):
                # action_probs is baked into reachprobs for everyone except the acting player
                if player == root.player:
                    payoffs[player] += winnings * action_probs[i]
                else:
                    payoffs[player] += winnings
        # Update regret calculations
        self.cfr_regret_update(root, action_payoffs, payoffs[root.player])
        return payoffs

    def cfr_strategy_update(self, root, reachprobs):
        # Update the strategies and regrets for each infoset
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
        # Get the current CFR
        prev_cfr = self.counterfactual_regret[root.player][infoset]
        # Get the total positive CFR
        sumpos_cfr = float(sum([max(0, x) for x in prev_cfr]))
        if sumpos_cfr == 0:
            # Default strategy is equal probability
            probs = self.equal_probs(root)
        else:
            # Use the strategy that's proportional to accumulated positive CFR
            probs = [max(0, x) / sumpos_cfr for x in prev_cfr]
        # Use the updated strategy as our current strategy
        self.current_profile.strategies[root.player].policy[infoset] = probs
        # Update the weighted policy probabilities (used to recover the average strategy)
        for i in range(3):
            self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player] * probs[i]
        if sum(self.action_reachprobs[root.player][infoset]) == 0:
            # Default strategy is equal weight
            self.profile.strategies[root.player].policy[infoset] = self.equal_probs(root)
        else:
            # Recover the weighted average strategy
            self.profile.strategies[root.player].policy[infoset] = [
                self.action_reachprobs[root.player][infoset][i] / sum(self.action_reachprobs[root.player][infoset]) for
                i in range(3)]
        # Return and use the current CFR strategy
        return self.current_profile.strategies[root.player]

    def cfr_regret_update(self, root, action_payoffs, ev):
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        for i, subpayoff in enumerate(action_payoffs):
            if subpayoff is None:
                continue
            immediate_cfr = subpayoff[root.player] - ev
            infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
            self.counterfactual_regret[root.player][infoset][i] += immediate_cfr


class OutcomeSamplingCFR(ChanceSamplingCFR):
    def __init__(self, rules, exploration=0.4):
        ChanceSamplingCFR.__init__(self, rules)
        self.exploration = exploration

    def cfr(self):
        # Sample all cards to be used
        holecards_per_player = sum([x.holecards for x in self.rules.roundinfo])
        boardcards_per_hand = sum([x.boardcards for x in self.rules.roundinfo])
        todeal = random.sample(self.rules.deck, boardcards_per_hand + holecards_per_player * self.rules.players)
        # Deal holecards
        self.holecards = [tuple(todeal[p * holecards_per_player:(p + 1) * holecards_per_player]) for p in
                          range(self.rules.players)]
        self.board = tuple(todeal[-boardcards_per_hand:])
        # Set the top card of the deck
        self.top_card = len(todeal) - boardcards_per_hand
        # Call the standard CFR algorithm
        self.cfr_helper(self.tree.root, [1 for _ in range(self.rules.players)], 1.0)

    def cfr_helper(self, root, reachprobs, sampleprobs):
        if type(root) is TerminalNode:
            return self.cfr_terminal_node(root, reachprobs, sampleprobs)
        if type(root) is HolecardChanceNode:
            return self.cfr_holecard_node(root, reachprobs, sampleprobs)
        if type(root) is BoardcardChanceNode:
            return self.cfr_boardcard_node(root, reachprobs, sampleprobs)
        return self.cfr_action_node(root, reachprobs, sampleprobs)

    def cfr_terminal_node(self, root, reachprobs, sampleprobs):
        payoffs = [0 for _ in range(self.rules.players)]
        for hands, winnings in root.payoffs.items():
            if not self.terminal_match(hands):
                continue
            for player in range(self.rules.players):
                prob = 1.0
                for opp, hc in enumerate(hands):
                    if opp != player:
                        prob *= reachprobs[opp]
                payoffs[player] = prob * winnings[player] / sampleprobs
            return payoffs

    def cfr_holecard_node(self, root, reachprobs, sampleprobs):
        assert (len(root.children) == 1)
        return self.cfr_helper(root.children[0], reachprobs, sampleprobs)

    def cfr_boardcard_node(self, root, reachprobs, sampleprobs):
        # Number of community cards dealt this round
        num_dealt = len(root.children[0].board) - len(root.board)
        # Find the child that matches the sampled board card(s)
        for bc in root.children:
            if self.boardmatch(num_dealt, bc):
                # Perform normal CFR
                results = self.cfr_helper(bc, reachprobs, sampleprobs)
                # Return the payoffs
                return results
        raise Exception('Sampling from impossible board card')

    def cfr_action_node(self, root, reachprobs, sampleprobs):
        # Calculate strategy from counterfactual regret
        strategy = self.cfr_strategy_update(root, reachprobs, sampleprobs)
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
        action_probs = strategy.probs(infoset)
        if random.random() < self.exploration:
            action = self.random_action(root)
        else:
            action = strategy.sample_action(infoset)
        reachprobs[root.player] *= action_probs[action]
        csp = self.exploration * (1.0 / len(root.children)) + (1.0 - self.exploration) * action_probs[action]
        payoffs = self.cfr_helper(root.get_child(action), reachprobs, sampleprobs * csp)
        # Update regret calculations
        self.cfr_regret_update(root, payoffs[root.player], action, action_probs[action])
        payoffs[root.player] *= action_probs[action]
        return payoffs

    def random_action(self, root):
        options = []
        if root.fold_action:
            options.append(FOLD)
        if root.call_action:
            options.append(CALL)
        if root.raise_action:
            options.append(RAISE)
        return random.choice(options)

    def cfr_strategy_update(self, root, reachprobs, sampleprobs):
        # Update the strategies and regrets for each infoset
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
        # Get the current CFR
        prev_cfr = self.counterfactual_regret[root.player][infoset]
        # Get the total positive CFR
        sumpos_cfr = float(sum([max(0, x) for x in prev_cfr]))
        if sumpos_cfr == 0:
            # Default strategy is equal probability
            probs = self.equal_probs(root)
        else:
            # Use the strategy that's proportional to accumulated positive CFR
            probs = [max(0, x) / sumpos_cfr for x in prev_cfr]
        # Use the updated strategy as our current strategy
        self.current_profile.strategies[root.player].policy[infoset] = probs
        # Update the weighted policy probabilities (used to recover the average strategy)
        for i in range(3):
            self.action_reachprobs[root.player][infoset][i] += reachprobs[root.player] * probs[i] / sampleprobs
        if sum(self.action_reachprobs[root.player][infoset]) == 0:
            # Default strategy is equal weight
            self.profile.strategies[root.player].policy[infoset] = self.equal_probs(root)
        else:
            # Recover the weighted average strategy
            self.profile.strategies[root.player].policy[infoset] = [
                self.action_reachprobs[root.player][infoset][i] / sum(self.action_reachprobs[root.player][infoset]) for
                i in range(3)]
        # Return and use the current CFR strategy
        return self.current_profile.strategies[root.player]

    def cfr_regret_update(self, root, ev, action, actionprob):
        hc = self.holecards[root.player][0:len(root.holecards[root.player])]
        infoset = self.rules.infoset_format(root.player, hc, root.board, root.bet_history)
        for i in range(3):
            if not root.valid(i):
                continue
            immediate_cfr = -ev * actionprob
            if action == i:
                immediate_cfr += ev
            self.counterfactual_regret[root.player][infoset][i] += immediate_cfr


def near(val, expected, distance=0.0001):
    return val >= (expected - distance) and val <= (expected + distance)


if __name__ == "__main__":
    print('')
    print('')
    print('Testing CFR')
    print('')
    print('')
    print('Computing NE for Half-Street Kuhn poker')

    hskuhn = half_street_kuhn_rules()  # 游戏设置
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
    # print('Verifying P1 policy')
    # assert (near(cfr.profile.strategies[0].policy['Q:/:'][CALL], 2.0 / 3.0, 0.01))
    # assert (near(cfr.profile.strategies[0].policy['Q:/:'][RAISE], 1.0 / 3.0, 0.01))
    # assert (near(cfr.profile.strategies[0].policy['K:/:'][CALL], 1, 0.01))
    # assert (near(cfr.profile.strategies[0].policy['K:/:'][RAISE], 0, 0.01))
    # assert (near(cfr.profile.strategies[0].policy['A:/:'][CALL], 0, 0.01))
    # assert (near(cfr.profile.strategies[0].policy['A:/:'][RAISE], 1.0, 0.01))
    # print('Verifying P2 policy')
    # assert (near(cfr.profile.strategies[1].policy['Q:/r:'][FOLD], 1.0, 0.01))
    # assert (near(cfr.profile.strategies[1].policy['Q:/r:'][CALL], 0, 0.01))
    # assert (near(cfr.profile.strategies[1].policy['K:/r:'][FOLD], 2.0 / 3.0, 0.01))
    # assert (near(cfr.profile.strategies[1].policy['K:/r:'][CALL], 1.0 / 3.0, 0.01))
    # assert (near(cfr.profile.strategies[1].policy['A:/r:'][FOLD], 0, 0.01))
    # assert (near(cfr.profile.strategies[1].policy['A:/r:'][CALL], 1.0, 0.01))
    #
    # print('Done!')
    # print('')

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
