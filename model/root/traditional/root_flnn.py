from utils.MathUtil import *
from utils.IOUtil import *
from model.root.root_base import RootBase
import time

class RootFlnn(RootBase):
    def __init__(self, root_base_paras=None, root_flnn_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.activation = root_flnn_paras["activation"]
        self.epoch = root_flnn_paras["epoch"]
        self.lr = root_flnn_paras["lr"]
        self.batch_size = root_flnn_paras["batch_size"]
        self.beta = root_flnn_paras["beta"]

        if self.activation == "self":
            self.activation_function = itself
            self.activation_backward = derivative_self
        elif self.activation == "elu":
            self.activation_function = elu
            self.activation_backward = derivative_elu
        elif self.activation == "relu":
            self.activation_function = relu
            self.activation_backward = derivative_relu
        elif self.activation == "tanh":
            self.activation_function = tanh
            self.activation_backward = derivative_tanh
        elif self.activation == "sigmoid":
            self.activation_function = sigmoid
            self.activation_backward = derivative_sigmoid

    def _forecasting__(self):
        # Evaluate models on the test set
        y_pred = self.activation_function( np.dot(self.X_test, self.model["w"]) + self.model["b"] )
        pred_inverse = self.time_series.minmax_scaler.inverse_transform(y_pred)
        real_inverse = self.time_series.minmax_scaler.inverse_transform(self.y_test)
        return real_inverse, pred_inverse, self.y_test, y_pred

    def _running__(self):
        self.time_system = time.time()
        self._preprocessing__()
        self.time_total_train = time.time()
        self._training__()
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

