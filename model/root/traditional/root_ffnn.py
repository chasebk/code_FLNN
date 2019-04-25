from model.root.root_base import RootBase
import time

class RootFfnn(RootBase):
    """
    Root of all traditional feed-forward neural network (Multi Layer Neural Network)
    """
    def __init__(self, root_base_paras=None, root_ann_paras=None):
        RootBase.__init__(self, root_base_paras)
        self.hidden_sizes = root_ann_paras["hidden_sizes"]
        self.epoch = root_ann_paras["epoch"]
        self.batch_size = root_ann_paras["batch_size"]
        self.lr = root_ann_paras["lr"]
        self.activations = root_ann_paras["activations"]
        self.optimizer = root_ann_paras["optimizer"]
        self.loss = root_ann_paras["loss"]

    def _forecasting__(self):
        # Evaluate models on the test set
        y_pred = self.model.predict(self.X_test)
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

