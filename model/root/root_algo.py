import numpy as np
from copy import deepcopy

class RootAlgo(object):
    """
    This is root of all Natural Inspired Algorithms
    """
    ID_MIN_PROBLEM = 0
    ID_MAX_PROBLEM = -1

    ID_ERROR = 0        # MSE = 0, MAE = 1

    def __init__(self, root_algo_paras = None):
        self.X_train = root_algo_paras["X_train"]
        self.y_train = root_algo_paras["y_train"]
        self.X_valid = root_algo_paras["X_valid"]
        self.y_valid = root_algo_paras["y_valid"]
        self.problem_size = root_algo_paras["problem_size"]
        self.train_valid_rate = root_algo_paras["train_valid_rate"]
        self.domain_range = root_algo_paras["domain_range"]
        self.print_train = root_algo_paras["print_train"]
        self._get_average_error__ = root_algo_paras["_get_average_error__"]
        self.solution, self.loss_train = None, []

    def _create_solution__(self, minmax=0):
        solution = np.random.uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(model=solution, minmax=minmax)
        return [solution, fitness]

    def _fitness_model__(self, model=None, minmax=0):
        """ distance between the sum of an individual numbers and the target number. Lower is better"""
        ae_train = self._get_average_error__(model, self.X_train, self.y_train)
        ae_valid = self._get_average_error__(model, self.X_valid, self.y_valid)
        mse = self.train_valid_rate[0] * ae_train[0] + self.train_valid_rate[1] * ae_valid[0]
        mae = self.train_valid_rate[0] * ae_train[1] + self.train_valid_rate[1] * ae_valid[1]
        return [mse, mae] if minmax == 0 else [1.0 / mse, 1.0 / mae]

    def _fitness_encoded__(self, encoded=None, id_pos=None, minmax=0):
        return self._fitness_model__(model=encoded[id_pos], minmax=minmax)

    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness][self.ID_ERROR])
        return deepcopy(sorted_pop[id_best])

    def _train__(self):
        pass

