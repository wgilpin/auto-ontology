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
        fn: Callable,
        n_runs: int = 1,
        sample_size:int = 3000,
        verbose: int=0) -> list[dict]:
    """
    Grid search for hyperparameter tuning.
    :param config: dict, hyperparameter configuration
        - key: hyperparameter name
        - value: list of hyperparameter values
    :param fn: callable, function to be evaluated, with params:
        : param config: dict, hyperparameter configuration for this run
        : param fn: callable, function to be evaluated
        : param index: int, index of this run
        : param sample_size: int, number of samples to be generated
        : param n_runs: int, number of runs for each hyperparameter configuration
        signature: fn(config: dict, index: int) -> None
    :param n: int, number of times to repeat the evaluation
    """
    # replace any single values with a list of length 1
    config = {k: [v] if not isinstance(v, list) else v for k, v in config.items()}
    # get all possible combinations of hyperparameter values
    params = list(config.keys())

    combos = [
        {params[i]: v for (i,v) in enumerate(x) }\
                                        for x in it.product(*config.values())]

    total_runs = math.prod([len(p_list) for p_list in config.values()])
    scores = []
    for index, combo in enumerate(combos * n_runs):
        scores.append(fn(combo, index, total_runs, sample_size, verbose))

    return scores
