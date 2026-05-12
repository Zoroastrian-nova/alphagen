import json
import sys

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.stock_data import StockData


def _load_config(config_path: str = "symbol_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


config = _load_config()
dso_cfg = config["dso"]

funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(dso_cfg["features"]):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in dso_cfg["constants"]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)


instruments = config["instruments"]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
reseed_everything(seed)

cache = {}
device = torch.device(config["device"])
dso_data = dso_cfg["data"]
data_train = StockData(instruments, dso_data["train_start"], dso_data["train_end"], device=device)
data_valid = StockData(instruments, dso_data["valid_start"], dso_data["valid_end"], device=device)
data_test = StockData(instruments, dso_data["test_start"], dso_data["test_end"], device=device)
calculator_train = QLibStockDataCalculator(data_train, target)
calculator_valid = QLibStockDataCalculator(data_valid, target)
calculator_test = QLibStockDataCalculator(data_test, target)


if __name__ == '__main__':
    X = np.array([dso_cfg["features"]])
    y = np.array([[1]])
    functions.function_map = funcs

    pool = MseAlphaPool(
        capacity=dso_cfg["pool_capacity"],
        calculator=calculator_train,
        ic_lower_bound=None
    )

    eval_interval = dso_cfg["eval_interval"]

    class Ev:
        def __init__(self, pool):
            self.cnt = 0
            self.pool: MseAlphaPool = pool
            self.results = {}

        def alpha_ev_fn(self, key):
            expr = eval(key)
            try:
                ret = self.pool.try_new_expr(expr)
            except OutOfDataRangeError:
                ret = -1.
            finally:
                self.cnt += 1
                if self.cnt % eval_interval == 0:
                    test_ic = pool.test_ensemble(calculator_test)[0]
                    self.results[self.cnt] = test_ic
                    print(self.cnt, test_ic)
                return ret

    ev = Ev(pool)

    dso_training = dso_cfg["training"]
    dso_prior = dso_cfg["prior"]
    model_config = dict(
        task=dict(
            task_type='regression',
            function_set=list(funcs.keys()),
            metric='alphagen',
            metric_params=[lambda key: ev.alpha_ev_fn(key)],
        ),
        training={
            'n_samples': dso_training["n_samples"],
            'batch_size': dso_training["batch_size"],
            'epsilon': dso_training["epsilon"],
        },
        prior={
            'length': {
                'min_': dso_prior["length_min"],
                'max_': dso_prior["length_max"],
                'on': dso_prior["on"],
            }
        }
    )

    # Create the model
    model = DeepSymbolicRegressor(config=model_config)
    model.fit(X, y)

    print(ev.results)
