import math
import itertools as it
from typing import Callable

def make_name(config: dict) -> str:
    """
    Create a unique name for a hyperparameter configuration.
    :param config: dict, hyperparameter configuration
    """
    return '_'.join([f'{k}-{v}' for k, v in config.items()])    

def grid_search(
        config: dict,
        fn: callable,
        n: int = 1,
        sample_size:int = 3000) -> list[dict]:
    """
    Grid search for hyperparameter tuning.
    :param config: dict, hyperparameter configuration
        - key: hyperparameter name
        - value: list of hyperparameter values
    :param fn: callable, function to be evaluated, with params:
        : param config: dict, hyperparameter configuration for this run
        : param index: int, index of this run
        signature: fn(config: dict, index: int) -> None
    :param n: int, number of times to repeat the evaluation
    """
    params = list(config.keys())

    combos = [
        {params[i]: v for (i,v) in enumerate(x) }\
                                        for x in it.product(*config.values())]

    n_runs = math.prod([len(p_list) for p_list in config.values()])
    scores = []
    for index, combo in enumerate(combos * n):
        scores.append(fn(combo, index, n_runs, sample_size))

    return scores
