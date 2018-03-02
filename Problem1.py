import numpy as np
import scr.FigureSupport as figureLibrary
import scr.StatisticalClasses as Stat

class Game(object):
    def __init__(self, id, prob_head):
        self._id = id
        self._rnd = np.random
        self._rnd.seed(id)
        self._probHead = prob_head  # probability of flipping a head
        self._countWins = 0  # number of wins, set to 0 to begin

    def simulate(self, n_of_flips):

        count_tails = 0  # number of consecutive tails so far, set to 0 to begin

        # flip the coin 20 times
        for i in range(n_of_flips):

            # in the case of flipping a heads
            if self._rnd.random_sample() < self._probHead:
                if count_tails >= 2:  # if the series is ..., T, T, H
                    self._countWins += 1  # increase the number of wins by 1
                count_tails = 0  # the tails counter needs to be reset to 0 because a heads was flipped

            # in the case of flipping a tails
            else:
                count_tails += 1  # increase tails count by one

    def get_reward(self):
        # calculate the reward from playing a single game
        return 100*self._countWins - 250

    def get_casino_reward(self):
        return 250 - 100*self._countWins


class SetOfGames:
    def __init__(self, prob_head, n_games):
        self._gameRewards = [] # create an empty list where rewards will be stored
        self._probLoss = []
        self._casinoRewards = []
        self._n_games = n_games

        # simulate the games
        for n in range(n_games):
            # create a new game
            game = Game(id=n, prob_head=prob_head)
            # simulate the game with 20 flips
            game.simulate(20)
            # store the reward
            self._gameRewards.append(game.get_reward())
            self._casinoRewards.append(game.get_casino_reward())

    def get_ave_reward(self):
        """ returns the average reward from all games"""
        return sum(self._gameRewards) / len(self._gameRewards)

    def get_ave_casino_reward(self):
        return sum(self._casinoRewards) / len(self._casinoRewards)

    def get_reward_list(self):
        """ returns all the rewards from all game to later be used for creation of histogram """
        return self._gameRewards

    def get_casino_reward_list(self):
        return self._casinoRewards

    def get_max(self):
        """ returns maximum reward"""
        return max(self._gameRewards)

    def get_min(self):
        """ returns minimum reward"""
        return min(self._gameRewards)

    def get_probability_loss(self):
        """ returns the probability of a loss """
        count_loss = 0
        for value in self._gameRewards:
            if value < 0:
                count_loss += 1
        probability_loss = count_loss / len(self._gameRewards)
        return probability_loss

    def get_probloss_list(self):
        for n in range(self._n_games):
            self._probLoss.append(self.get_probability_loss())
        return self._probLoss


class GameOutcomes:
    def __init__(self, simulated_game):
        self._simulatedGame = simulated_game
        self._sumStat_gameRewards = \
            Stat.SummaryStat('Game rewards', self._simulatedGame.get_reward_list())
        self._sumStat_probLoss = \
            Stat.SummaryStat('Probability of loss', self._simulatedGame.get_probloss_list())
        self._sumStat_casinoRewards = \
            Stat.SummaryStat('Casino rewards', self._simulatedGame.get_casino_reward_list())

    def get_ave_reward(self):
         return self._sumStat_gameRewards.get_mean()

    def get_CI_reward(self, alpha):
        return self._sumStat_gameRewards.get_t_CI(alpha)

    def get_ave_casino_reward(self):
        return self._sumStat_casinoRewards.get_mean()

    def get_CI_reward_Casino(self, alpha):
        return self._sumStat_casinoRewards.get_t_CI(alpha)

    def get_ave_prob_loss(self):
        return self._sumStat_probLoss.get_mean()

    def get_CI_prob_loss(self, alpha):
        return self._sumStat_probLoss.get_t_CI(alpha)


class MultiGame:
    def __init__(self, ids, n_games, prob_heads):
        self._ids = ids
        self._nGames = n_games
        self._probHeads = prob_heads
        self._expectedRewards = [] #2D list of expected rewards from each simulated game
        self._meanExpectedRewards = []
        self._sumStat_meanExpectedReward = None

    def simulate(self, n_of_flips):
        for i in range(len(self._ids)):
            # create game
            game1 = Game(self._ids[i], self._probHeads[i])
            # simulate game
            output = game1.simulate(n_of_flips)
            # store average rewards for this game
            self._expectedRewards.append(game.get_reward())
            # store average rewards for this game
            self._meanExpectedRewards.append(output.get)

        self._sumStat_meanExpectedReward = Stat.SummaryStat('Mean expected rewards', self._meanExpectedRewards)

    def get_games_mean_rewards(self, game_index):
        return self._meanExpectedRewards[game_index]

    def get_games_CI_mean_rewards(self, game_index, alpha):
        st = Stat.SummaryStat('', self._expectedRewards[game_index])
        return st.get_t_CI(alpha)

    def get_all_mean_rewards(self):
        return self._meanExpectedRewards

    def get_overall_mean_rewards(self):
        return self._sumStat_meanExpectedReward.get_

    def get_game_PI_reward(self, game_index, alpha):
        st = Stat.SummaryStat('', self._expectedRewards[game_index])
        return st.get_PI(alpha)

    def get_PI_mean_reward(self, alpha):
        return self._sumStat_meanExpectedReward.get_PI(alpha)


# Calculate expected reward of 1000 games
trial = SetOfGames(prob_head=0.5, n_games=1000)
print("The average expected reward is:", trial.get_ave_reward())

# minimum reward is -$250 if {T, T, H} never occurs.
# maximum reward is $250 if {T, T, H} occurs 6 times (if you increase the number of games you might see this outcome).

# find minimum and maximum reward in trial
print("In our trial, the maximum reward is:", trial.get_max())
print("In our trial, the minimum reward is:", trial.get_min())

# Find the probability of a loss
print("The probability of a single game yielding a loss is:", trial.get_probability_loss())


# PROBLEM 1 - Print the 95% t-based confidence intervals for the
# expected reward and probability of loss. You can use 1,000 simulated
# games to calculate these confidence intervals.

# Note - this would be steady-state simulation because the number of
# observations can fall under the Law of Large Numbers

ALPHA = 0.05
print("PROBLEM 1 ANSWERS:")
print("95% CI of average expected reward is:", GameOutcomes(simulated_game=trial).get_CI_reward(ALPHA))
print("95% CI of probability of loss is:", GameOutcomes(simulated_game=trial).get_CI_prob_loss(ALPHA))

Casino = SetOfGames(prob_head=0.5, n_games=1000)
print("PROBLEM 3 ANSWERS:")
print("For a casino owner playing the game 1,000 times, the average expected "
      "reward for the casino owner is:", Casino.get_ave_casino_reward())
print("For the casino owner, the 95% CI of "
      "the casino's average expected reward is:",GameOutcomes(simulated_game=Casino).get_CI_reward_Casino(ALPHA))

Gambler = MultiGame(ids=1, n_games=10, prob_heads=0.5)
print("For a gambler playing the game 10 times, the average expected "
      "reward is:", Gambler.get_overall_mean_rewards())
