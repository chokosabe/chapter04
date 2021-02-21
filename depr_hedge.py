import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


sl = -0.03
tp = 0.1

class Actions(Enum):
    Sell = 1
    Hold = 0
    Buy = 2


class Positions(Enum):
    Short = 1
    Hold = 0
    Long = 2
    matched_actions = {
        Actions.Sell.value: Short,
        Actions.Hold.value: Hold,
        Actions.Buy.value: Long
    }

class HedgeEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config: Dict[] = env_config):

        self.trade_fee = 0.04  # unit

        self.df = env_config['df']
        self.window_size = env_config['window_size']
        self.frame_bound = env_config['frame_bound']

        self.seed()
        self.prices, self.signal_features = self._process_data()
        self.shape = (self.window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Hold.value
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._interim_reward = 0.
        self._total_profit = 0.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def apply_trade_fee(self):
        self._interim_reward -= self.trade_fee
        self._total_reward -= self.trade_fee

    def step(self, action):
        step_reward = 0
        action = int(action)
        self._done = False

        if self._current_tick >= self._end_tick:
            self._done = True

        if self._done is False:
            step_reward = self._calculate_reward(action)

        self._current_tick += 1

        self._total_reward += step_reward
        self._interim_reward += step_reward
        self._update_profit(step_reward)
        trade = self.trade_condition(action)

        if trade:
            self._position = Positions.matched_actions.value[action]
            if self._position in [Positions.Short.value, Positions.Long.value]:
                self.apply_trade_fee()
                step_reward -= self.trade_fee

        #if self._interim_reward < sl:
        #    self._position = Positions.Hold.value
        #    self._interim_reward = 0

        #if self._interim_reward > tp:
        #    self._position = Positions.Hold.value
        #    self._interim_reward = 0

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position
        )
        self._update_history(info)

        self._reward = step_reward
        if self._done is True:
            self._reward = self._total_reward

        return observation, self._reward, self._done, info


    def _get_observation(self):

        start_tick = self._current_tick - self.window_size
        return self.signal_features[start_tick:self._current_tick]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short.value:
                color = 'red'
            elif position == Positions.Long.value:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short.value:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long.value:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )


    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()

    def trade_condition(self, action):
        return (
                (action == Actions.Buy.value and self._position in [Positions.Short.value, Positions.Hold.value]) or
                (action == Actions.Sell.value and self._position in [Positions.Long.value, Positions.Hold.value]) or
                (action == Actions.Hold.value and self._position in [Positions.Long.value, Positions.Short.value])
        )

    def _process_data(self):
        prices = self.df.loc[:, 'prices'].to_numpy()
        normalised_prices = self.df.loc[:, 'normalised_prices'].to_numpy()
        ts_sin = self.df.loc[:, 'timestamp_sin'].to_numpy()
        ts_cos = self.df.loc[:, 'timestamp_cos'].to_numpy()

        #prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        normalised_prices = normalised_prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        ts_sin = ts_sin[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        ts_cos = ts_sin[self.frame_bound[0]-self.window_size:self.frame_bound[1]]


        #diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((normalised_prices, ts_sin, ts_cos ))

        return prices, signal_features

    def _calculate_reward(self, action):
        """
        We calculate a step reward if the preceding position (going into the step) was either long or short
        """
        price_diff = 0
        step_reward = 0

        trade = self.trade_condition(action)

        #if trade is True:
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        if self._position == Positions.Long.value:
            price_diff = (current_price/last_trade_price) - 1

        if self._position == Positions.Short.value:
            price_diff = 1 - (current_price/last_trade_price)
        price_diff = price_diff * 100.00
        step_reward += price_diff
        self._last_trade_tick = self._current_tick
        return step_reward


    def _update_profit(self, step_reward):
        #trade = self.trade_condition(action)
        self._total_profit += step_reward

    # if trade or self._done:
    #     current_price = self.prices[self._current_tick]
    #     last_trade_price = self.prices[self._last_trade_tick]

    #     if self._position == Positions.Long:
    #         profit = self.total_profit - trade_fee
    #         shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
    #         self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price


#    def max_possible_profit(self):
#        current_tick = self._start_tick
#        last_trade_tick = current_tick - 1
#        profit = 1.
#
#        while current_tick <= self._end_tick:
#            position = None
#            if self.prices[current_tick] < self.prices[current_tick - 1]:
#                while (current_tick <= self._end_tick and
#                       self.prices[current_tick] < self.prices[current_tick - 1]):
#                    current_tick += 1
#                position = Positions.Short
#            else:
#                while (current_tick <= self._end_tick and
#                       self.prices[current_tick] >= self.prices[current_tick - 1]):
#                    current_tick += 1
#                position = Positions.Long
#
#            if position == Positions.Long:
#                current_price = self.prices[current_tick - 1]
#                last_trade_price = self.prices[last_trade_tick]
#                shares = profit / last_trade_price
#                profit = shares * current_price
#            last_trade_tick = current_tick - 1
#
#        return profit