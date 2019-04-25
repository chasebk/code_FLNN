import numpy as np
import time
from model.root.root_base import RootBase
from utils.MathUtil import itself, elu, relu, tanh, sigmoid
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RootHybridFlnn(RootBase):
    """
        This is root of all hybrid models which include Multi-layer Neural Network and Optimization Algorithms.
    """
    def __init__(self, root_base_paras=None, root_hybrid_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.epoch = root_hybrid_paras["epoch"]
        self.activation = root_hybrid_paras["activation"]
        self.train_valid_rate = root_hybrid_paras["train_valid_rate"]
        self.domain_range = root_hybrid_paras["domain_range"]

        if self.activation == "self":
            self._activation__ = itself
        elif self.activation == "elu":
            self._activation__ = elu
        elif self.activation == "relu":
            self._activation__ = relu
        elif self.activation == "tanh":
            self._activation__ = tanh
        elif self.activation == "sigmoid":
            self._activation__ = sigmoid

    def _setting__(self):
        self.input_size, self.output_size = self.X_train.shape[1], self.y_train.shape[1]
        self.w_size = self.input_size * self.output_size
        self.b_size = self.output_size
        self.problem_size = self.w_size + self.b_size
        self.root_algo_paras = {
            "X_train": self.X_train, "y_train": self.y_train, "X_valid": self.X_valid, "y_valid": self.y_valid,
            "train_valid_rate": self.train_valid_rate, "domain_range": self.domain_range,
            "problem_size": self.problem_size, "print_train": self.print_train,
            "_get_average_error__": self._get_average_error__
        }

    def _training__(self):
        pass

    def _forecasting__(self):
        y_pred = self._activation__(np.add(np.matmul(self.X_test, self.model["w"]), self.model["b"]))
        real_inverse = self.time_series.minmax_scaler.inverse_transform(self.y_test)
        pred_inverse = self.time_series.minmax_scaler.inverse_transform(np.reshape(y_pred, self.y_test.shape))
        return real_inverse, pred_inverse, self.y_test, y_pred

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing__()
        self._setting__()
        self.time_total_train = time.time()
        self._training__()
        self._get_model__(self.solution)
        self.time_total_train = round(time.time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time.time()
        y_actual, y_predict, y_actual_normalized, y_predict_normalized = self._forecasting__()
        self.time_predict = round(time.time() - self.time_predict, 6)
        self.time_system = round(time.time() - self.time_system, 4)
        if self.test_type == "normal":
            self._save_results__(y_actual, y_predict, y_actual_normalized, y_predict_normalized, self.loss_train)
        elif self.test_type == "stability":
            self._save_results_ntimes_run__(y_actual, y_predict, y_actual_normalized, y_predict_normalized)


    ## Helper functions
    def _get_model__(self, individual=None):
        w = np.reshape(individual[:self.w_size], (self.input_size, self.output_size))
        b = np.reshape(individual[self.w_size:], (-1, self.output_size))
        self.model = {"w": w, "b": b}

    def _get_average_error__(self, individual=None, X_data=None, y_data=None):
        w = np.reshape(individual[:self.w_size], (self.input_size, self.output_size))
        b = np.reshape(individual[self.w_size:], (-1, self.output_size))
        y_pred = self._activation__(np.add(np.matmul(X_data, w), b))
        return [mean_squared_error(y_pred, y_data), mean_absolute_error(y_pred, y_data)]
