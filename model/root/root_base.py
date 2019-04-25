
from utils.PreprocessingUtil import TimeSeries
from utils.MeasureUtil import MeasureTimeSeries
from utils.IOUtil import save_all_models_to_csv, save_prediction_to_csv, save_loss_train_to_csv
from utils.GraphUtil import draw_predict_with_error

class RootBase(object):
    """
        This is root of all networks.
    """
    def __init__(self, root_base_paras=None):

        self.dataset = root_base_paras["dataset"]
        self.data_idx = root_base_paras["data_idx"]
        self.sliding = root_base_paras["sliding"]
        self.expand_function = root_base_paras["expand_function"]
        self.output_idx = root_base_paras["output_idx"]
        self.method_statistic = root_base_paras["method_statistic"]

        self.path_save_result = root_base_paras["path_save_result"]
        self.log_filename = root_base_paras["log_filename"]
        self.multi_output = root_base_paras["multi_output"]
        self.test_type = root_base_paras["test_type"]
        self.draw = root_base_paras["draw"]
        self.print_train = root_base_paras["print_train"]

        self.time_series = None
        self.model, self.solution, self.loss_train, self.filename = None, None, [], None
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None
        self.time_total_train, self.time_epoch, self.time_predict, self.time_system = None, None, None, None

    def _preprocessing__(self):
        self.time_series = TimeSeries(self.dataset, self.data_idx, self.sliding, self.expand_function, self.output_idx, self.method_statistic)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.time_series._preprocessing_2d__()

    def _preprocessing_3d__(self):
        self.time_series = TimeSeries(self.dataset, self.data_idx, self.sliding, self.expand_function, self.output_idx,self.method_statistic)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self.time_series._preprocessing_3d__()

    def _save_results__(self, y_actual=None, y_predict=None, y_actual_normalized=None, y_predict_normalized=None, loss_train=None):
        if self.multi_output:
            measure = MeasureTimeSeries(y_actual_normalized, y_predict_normalized, "raw_values", number_rounding=4)
            measure.fit()
            item = [self.filename, self.time_total_train, self.time_epoch, self.time_predict, self.time_system,
                    measure.score_ev[0], measure.score_mae[0], measure.score_mse[0], measure.score_msle[0],
                    measure.score_r2[0], measure.score_rmse[0], measure.score_mape[0], measure.score_smape[0],
                    measure.score_ev[1], measure.score_mae[1], measure.score_mse[1], measure.score_msle[1],
                    measure.score_r2[1], measure.score_rmse[1], measure.score_mape[1], measure.score_smape[1],
                    ]
            save_all_models_to_csv(item, self.log_filename, self.path_save_result)
            save_prediction_to_csv(y_actual[:,0:1], y_predict[:,0:1], self.filename, self.path_save_result+"CPU-")
            save_prediction_to_csv(y_actual[:,1:2], y_predict[:,1:2], self.filename, self.path_save_result+"RAM-")
            save_loss_train_to_csv(loss_train, self.filename, self.path_save_result + "Error-")
            if self.draw:
                draw_predict_with_error(1, [y_actual[:,0:1], y_predict[:,0:1]], [measure.score_rmse[0], measure.score_mae[0]], self.filename, self.path_save_result+"CPU-")
                draw_predict_with_error(2, [y_actual[:,1:2], y_predict[:,1:2]], [measure.score_rmse[1], measure.score_mae[1]], self.filename, self.path_save_result+"RAM-")
            if self.print_train:
                print('Predict DONE - CPU - RMSE: %f, RAM - RMSE: %f' % (measure.score_rmse[0], measure.score_rmse[1]))
        else:
            measure_nor = MeasureTimeSeries(y_actual_normalized, y_predict_normalized, None, number_rounding=4)
            measure_nor.fit()
            measure_denor = MeasureTimeSeries(y_actual, y_predict, None, number_rounding=4)
            measure_denor.fit()

            item = [self.filename, self.time_total_train, self.time_epoch, self.time_predict, self.time_system,
                    measure_nor.score_ev, measure_nor.score_mae, measure_nor.score_mse, measure_nor.score_msle,
                    measure_nor.score_r2, measure_nor.score_rmse, measure_nor.score_mape[0], measure_nor.score_smape[0],
                    measure_denor.score_ev, measure_denor.score_mae, measure_denor.score_mse, measure_denor.score_msle,
                    measure_denor.score_r2, measure_denor.score_rmse, measure_denor.score_mape[0], measure_denor.score_smape[0]
                    ]
            save_all_models_to_csv(item, self.log_filename, self.path_save_result)
            save_prediction_to_csv(y_actual, y_predict, self.filename, self.path_save_result)
            save_loss_train_to_csv(loss_train, self.filename, self.path_save_result + "Error-")
            if self.draw:
                draw_predict_with_error(1, [y_actual, y_predict], [measure_denor.score_rmse, measure_denor.score_mae], self.filename, self.path_save_result)
            if self.print_train:
                print('Predict DONE - RMSE: %f, MAE: %f' % (measure_denor.score_rmse, measure_denor.score_mae))

    def _save_results_ntimes_run__(self, y_actual=None, y_predict=None, y_actual_normalized=None, y_predict_normalized=None):
        measure_nor = MeasureTimeSeries(y_actual_normalized, y_predict_normalized, None, number_rounding=4)
        measure_nor.fit()
        measure_denor = MeasureTimeSeries(y_actual, y_predict, None, number_rounding=4)
        measure_denor.fit()
        item = [self.filename, self.time_total_train, self.time_epoch, self.time_predict, self.time_system,
                measure_nor.score_ev, measure_nor.score_mae, measure_nor.score_mse, measure_nor.score_msle,
                measure_nor.score_r2, measure_nor.score_rmse, measure_nor.score_mape[0], measure_nor.score_smape[0],
                measure_denor.score_ev, measure_denor.score_mae, measure_denor.score_mse, measure_denor.score_msle,
                measure_denor.score_r2, measure_denor.score_rmse, measure_denor.score_mape[0], measure_denor.score_smape[0]
                ]
        save_all_models_to_csv(item, self.log_filename, self.path_save_result)

    def _forecasting__(self):
        pass

    def _training__(self):
        pass

    def _running__(self):
        pass
