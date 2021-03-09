#!/usr/bin/env python
# Cryptocurrency trading RL environment with continuous-valued trade actions
# Chapter 4, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import os
import pprint
from typing import Dict

import gym
import numpy as np
import pandas as pd
from gym import spaces


class HedgeEnv(gym.Env):
    def __init__(self, env_config):
        """
        Hedge Training Environment
        """
        super().__init__()
        self.df = env_config["df"]
        self.opening_account_balance = env_config.get("opening_account_balance", 1.0)
        # Action: 1-dim value indicating a fraction amount of shares to Buy (0 to 1) or
        # sell (-1 to 0). The fraction is taken on the allowable number of
        # shares that can be bought or sold based on the account balance (no margin).
        self.action_space = spaces.Box(
            low=np.array([-1]), high=np.array([1]), dtype=np.float
        )

        self.observation_features = [
            # "prices",
            "normalised_prices",
            "timestamp_sin",
            "timestamp_cos",
        ]
        self.seed = env_config.get("seed")
        self.trade_cost = env_config.get("trade_cost", 0.005)
        self.stop_loss = env_config.get("stop_loss", 0.05)
        self.prices = self.df["prices"]
        self.horizon = env_config.get("horizon", len(self.df))
        self.window_size = env_config["window_size"]
        self.frame_bound = env_config.get("frame_bound", self.horizon)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.observation_features), self.window_size + 1),
            dtype=np.float,
        )
        self.order_size = env_config.get("order_size", 1)
        self.viz = None  # Visualizer
        # self._process_data()

    def step(self, action):
        # Execute one step within the trading environment
        self.execute_trade_action(action)

        self.current_step += 1

        reward = self.account_value - self.opening_account_balance  # Profit (loss)
        done = (self.horizon - (self.current_step + self.window_size)) <= 1
        obs = self.get_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cash_balance = self.opening_account_balance
        self.account_value = self.opening_account_balance
        self.current_step = 1
        self.trades = []
        self.previous_action = 0

        return self.get_observation()

    def render(self, **kwargs):
        # Render the environment to the screen
        pprint.pprint(self.trades[-5:])

    # def get_observation(self):
    #     return self.signal_features[
    #         self.current_step : self.current_step + self.window_size
    #     ]

    def get_observation(self):
        # Get price info data table from input (env_config) df
        observation = (
            self.df.loc[
                self.current_step : self.current_step + self.window_size,
                self.observation_features,
            ]
            .to_numpy()
            .T
        )
        return observation

    def execute_trade_action(self, action):
        if self.account_value < 0.5:
            # We dont have enough to continue
            return


        if action == 0:  # Indicates "Hold" action
            # Hold position; No trade to be executed
            # If already in a trade no trade cost is incurred
            # We just update the previous action value
            self.previous_action = "hold"
            return

        order_type = "buy" if action > 0 else "sell"
        
        transaction_cost = self._get_trade_cost(order_type)
        
        # Assign the previous price and current price
        current_price = self.df.loc[self.current_step, "prices"]
        previous_price = self.df.loc[self.current_step - 1, "prices"]

        if order_type == "buy":
            if self.previous_action == order_type:
                gain_diff = ((current_price - previous_price) / previous_price) * 100
            else:
                gain_diff = ((previous_price - current_price) / current_price) * 100

        elif order_type == "sell":

            if self.previous_action == order_type:
                gain_diff = ((previous_price - current_price) / current_price) * 100
            else:
                gain_diff = ((current_price - previous_price) / current_price) * 100

        # ordered_list = sorted(trades, key=lambda k: k['step'])
        # Take the last key
        # Loop through from the end
        # Find the index where the value changes
        # Sum between your starting and the end index
        # if this sum is less than the negative of the stop loss then we
        # have been stopped out

        # Update the previous action value
        self.previous_action = order_type

        # Extract any transaction costs
        self.account_value -= transaction_cost
        # Update account value with any trade gains or loss
        self.account_value += gain_diff

        self.trades.append(
            {
                "type": "sell",
                "step": self.current_step,
                "previous_action": self.previous_action,
                "trade_cost": transaction_cost,
                "gain diff": gain_diff,
                "account_value": self.account_value
            }
        )

        #self.render()

    def _get_trade_cost(self, action):
        if action != self.previous_action:
            return self.trade_cost
        return 0

    def _process_data(self):
        prices = self.df.loc[:, "prices"].to_numpy()
        normalised_prices = self.df.loc[:, "normalised_prices"].to_numpy()
        ts_sin = self.df.loc[:, "timestamp_sin"].to_numpy()
        ts_cos = self.df.loc[:, "timestamp_cos"].to_numpy()

        # prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[0 : self.horizon]
        normalised_prices = normalised_prices[0 : self.horizon]
        ts_sin = ts_sin[0 : self.horizon]
        ts_cos = ts_sin[0 : self.horizon]

        diff = np.insert(np.diff(prices), 0, 0)
        self.signal_features = np.column_stack((diff, ts_sin, ts_cos))
        self.prices = prices


#if __name__ == "__main__":
#    env = HedgeEnv()
#    obs = env.reset()
#    for _ in range(600):
#        action = env.action_space.sample()
#        next_obs, reward, done, _ = env.step(action)
#        env.render()
