import json
import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


def _load_config(config_path: str = "symbol_config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


config = _load_config()
gp_cfg = config["gp"]
data_cfg = config["data"]

funcs = [make_function(**func._asdict()) for func in generic_funcs]

instruments = config["instruments"]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else gp_cfg["seed"]
reseed_everything(seed)

cache = {}
device = torch.device(config["device"])
initialize_qlib(config["qlib_data_path"])
data_train = StockData(instruments, data_cfg["train_start"], data_cfg["train_end"], device=device)
data_test = StockData(instruments, data_cfg["test_segments"][0][0], data_cfg["test_segments"][-1][-1], device=device)
calculator_train = QLibStockDataCalculator(data_train, target)
calculator_test = QLibStockDataCalculator(data_test, target)

pool_capacity = gp_cfg["pool_capacity"]
pool = MseAlphaPool(
    capacity=pool_capacity,
    calculator=calculator_train,
    ic_lower_bound=None,
    l1_alpha=gp_cfg["l1_alpha"],
    device=device
)

max_token_len = gp_cfg["max_token_length"]

def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > max_token_len:
        return -1.

    expr = eval(key)
    try:
        ic = calculator_train.calc_single_IC_ret(expr)
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic


Metric = make_fitness(function=_metric, greater_is_better=True)


def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    expr = eval(top_key)
    ic_test, ric_test = calculator_test.calc_single_all_ret(expr)
    return {
        'ic_test': ic_test,
        'ric_test': ric_test
    }


def try_pool(capacity: int, mutual_ic_thres: Optional[float] = None):
    pool = MseAlphaPool(
        capacity=capacity,
        calculator=calculator_train,
        ic_lower_bound=None
    )
    exprs = []

    def acceptable(expr: str) -> bool:
        if mutual_ic_thres is None:
            return True
        return all(abs(pool.calculator.calc_mutual_IC(e, eval(expr))) <= mutual_ic_thres
                   for e in exprs)

    most_common = dict(Counter(cache).most_common(capacity if mutual_ic_thres is None else None))
    for key in most_common:
        if acceptable(key):
            exprs.append(eval(key))
            if len(exprs) >= capacity:
                break
    pool.force_load_exprs(exprs)

    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)
    return {
        "ic_train": ic_train,
        "ric_train": ric_train,
        "ic_test": ic_test,
        "ric_test": ric_test,
        "pool_state": pool.to_json_dict()
    }


generation = 0

def ev():
    global generation
    generation += 1
    directory = f"{config['paths']['gp_output']}/{seed}"
    os.makedirs(directory, exist_ok=True)
    if generation % gp_cfg["eval_generation_interval"] != 0:
        return
    capacity = gp_cfg["pool_capacity"]
    res = {"pool": capacity, "res": try_pool(capacity, mutual_ic_thres=gp_cfg["mutual_ic_thres"])}
    with open(f'{directory}/{generation}.json', 'w') as f:
        json.dump({'res': res, 'cache': cache}, f, indent=4)


if __name__ == '__main__':
    features = gp_cfg["features"]
    constants = [f'Constant({v})' for v in gp_cfg["constants"]]
    terminals = features + constants

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    sr_cfg = gp_cfg["symbolic_regressor"]
    est_gp = SymbolicRegressor(
        population_size=sr_cfg["population_size"],
        generations=sr_cfg["generations"],
        init_depth=tuple(sr_cfg["init_depth"]),
        tournament_size=sr_cfg["tournament_size"],
        stopping_criteria=sr_cfg["stopping_criteria"],
        p_crossover=sr_cfg["p_crossover"],
        p_subtree_mutation=sr_cfg["p_subtree_mutation"],
        p_hoist_mutation=sr_cfg["p_hoist_mutation"],
        p_point_mutation=sr_cfg["p_point_mutation"],
        p_point_replace=sr_cfg["p_point_replace"],
        max_samples=sr_cfg["max_samples"],
        verbose=1,
        parsimony_coefficient=sr_cfg["parsimony_coefficient"],
        random_state=seed,
        function_set=funcs,
        metric=Metric,  # type: ignore
        const_range=None,
        n_jobs=sr_cfg["n_jobs"]
    )
    est_gp.fit(X_train, y_train, callback=ev)
    print(est_gp._program.execute(X_train))
