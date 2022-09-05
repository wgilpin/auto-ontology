import math
import itertools as it

def grid_search(config: dict, fn: callable, n: int = 1) -> None:
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
        scores.append(fn(combo, index, n_runs))

    return scores
