from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout
from model.root.traditional.root_ffnn import RootFfnn

class Mlnn1HL(RootFfnn):
    """
    Multi-Layer Neural Network (1 Hidden Layer)
    """
    def __init__(self, root_base_paras=None, root_mlnn1hl_paras=None):
        RootFfnn.__init__(self, root_base_paras, root_mlnn1hl_paras)
        self.filename = "MLNN-1H-sliding_{}-net_para_{}".format(root_base_paras["sliding"], root_mlnn1hl_paras)

    def _training__(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        self.model.add(Dense(1, activation=self.activations[1]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        backend.set_session(backend.tf.Session(config=backend.tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads = 2)))
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]


class Mlnn2HL(RootFfnn):
    """
        Multi-Layer Neural Network (2 Hidden Layer)
        """
    def __init__(self, root_base_paras=None, root_mlnn2hl_paras=None):
        RootFfnn.__init__(self, root_base_paras, root_mlnn2hl_paras)
        self.filename = "MLNN-2H-sliding_{}-net_para_{}".format(root_base_paras["sliding"], root_mlnn2hl_paras)

    def _training__(self):
        self.model = Sequential()
        self.model.add(Dense(self.hidden_sizes[0], input_dim=self.X_train.shape[1], activation=self.activations[0]))
        self.model.add(Dense(self.hidden_sizes[1], activation=self.activations[1]))
        self.model.add(Dense(1, activation=self.activations[2]))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        backend.set_session(backend.tf.Session(
            config=backend.tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2)))
        ml = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=self.print_train)
        self.loss_train = ml.history["loss"]
