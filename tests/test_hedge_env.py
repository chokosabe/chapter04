import pandas as pd
import numpy as np

from hedge_env import HedgeEnv


class TestHedgeEnv:
    def test_initial(self):
        data = {
            "prices": [1, 2, 3, 4],
            "normalised_prices": [0.1, 0.2, 0.3, 0.4],
            "timestamp_sin": [0.11, 0.22, 0.33, 0.44],
            "timestamp_cos": [0.44, 0.33, 0.22, 0.11],
        }
        df = pd.DataFrame(data=data)
        env_config = {"df": df}
        env = HedgeEnv(env_config)
        expected = np.array(
            [
                # data["prices"],
                data["normalised_prices"][1:],
                data["timestamp_sin"][1:],
                data["timestamp_cos"][1:],
            ]
        )
        assert (env.reset() == expected).all()
        assert env.current_step == 1

    def test_train(self):
        data = {
            "prices": [0.001, 0.003, -0.001, 0.007],
            "normalised_prices": [0.1, 0.2, 0.3, 0.4],
            "timestamp_sin": [0.11, 0.22, 0.33, 0.44],
            "timestamp_cos": [0.44, 0.33, 0.22, 0.11],
        }
        df = pd.DataFrame(data=data)
        env_config = {"df": df}
        env = HedgeEnv(env_config)
        env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        assert done == False
        next_obs, reward, done, _ = env.step(action)
        assert done == True
        next_obs, reward, done, _ = env.step(action)
        assert done == True
