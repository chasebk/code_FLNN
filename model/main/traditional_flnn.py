from model.root.traditional.root_flnn import RootFlnn
import numpy as np
from utils.PreprocessingUtil import MiniBatch
from sklearn.metrics import mean_absolute_error, mean_squared_error

class FLNN(RootFlnn):
    def __init__(self, root_base_paras=None, root_flnn_paras=None):
        RootFlnn.__init__(self, root_base_paras, root_flnn_paras)
        self.filename = "FLNN-{}-nets_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]], root_flnn_paras)

    def _training__(self):
        number_input = self.X_train.shape[1]
        number_output = self.y_train.shape[1]

        ## init hyper and momentum parameters
        w, b = np.random.randn(number_input, number_output), np.zeros((1, number_output))
        vdw, vdb = np.zeros((number_input, number_output)), np.zeros((1, number_output))

        seed = 0
        for e in range(self.epoch):
            seed += 1
            mini_batches = MiniBatch(self.X_train, self.y_train, self.batch_size).random_mini_batches(seed=seed)

            total_error = 0
            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch
                X_batch, y_batch = X_batch.T, y_batch.T
                m = X_batch.shape[0]

                # Feed Forward
                z = np.add(np.matmul(X_batch, w), b)
                a = self.activation_function(z)

                total_error += mean_squared_error(a, y_batch)

                # Backpropagation
                da = a - y_batch
                dz = da * self.activation_backward(a)

                db = 1. / m * np.sum(dz, axis=0, keepdims=True)
                dw = 1. / m * np.matmul(X_batch.T, dz)

                vdw = self.beta * vdw + (1 - self.beta) * dw
                vdb = self.beta * vdb + (1 - self.beta) * db

                # Update weights
                w -= self.lr * vdw
                b -= self.lr * vdb
            self.loss_train.append(total_error / len(mini_batches))
            if self.print_train:
                print("> Epoch {0}: MSE {1}".format(e, total_error))
        self.model = {"w": w, "b": b}
