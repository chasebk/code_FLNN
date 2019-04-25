from model.optimizer.evolutionary import GA, DE
from model.optimizer.swarm import BFO, PSO, ABC, CSO
from model.root.hybrid.root_hybrid_flnn import RootHybridFlnn

class GaFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.ga_paras = ga_paras
        self.filename = "FL-GANN-{}-nets_{}-ga_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], ga_paras)

    def _training__(self):
        ga = GA.BaseGA(root_algo_paras=self.root_algo_paras, ga_paras = self.ga_paras)
        self.solution, self.loss_train = ga._train__()


class DeFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.de_paras = de_paras
        self.filename = "FL-DENN-{}-nets_{}-de_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], de_paras)

    def _training__(self):
        md = DE.BaseDE(root_algo_paras=self.root_algo_paras, de_paras = self.de_paras)
        self.solution, self.loss_train = md._train__()


class PsoFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.pso_paras = pso_paras
        self.filename = "FL-PSONN-{}-nets_{}-pso_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], pso_paras)

    def _training__(self):
        pso = PSO.BasePSO(root_algo_paras=self.root_algo_paras, pso_paras = self.pso_paras)
        self.solution, self.loss_train = pso._train__()


class BfoFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, bfo_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.bfo_paras = bfo_paras
        self.filename = "FL-BFONN-{}-nets_{}-bfo_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], bfo_paras)

    def _training__(self):
        md = BFO.BaseBFO(root_algo_paras=self.root_algo_paras, bfo_paras = self.bfo_paras)
        self.solution, self.loss_train = md._train__()


class ABfoLSFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abfols_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.abfols_paras = abfols_paras
        self.filename = "FL-ABFOLSNN-{}-nets_{}-abfols_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]],abfols_paras)

    def _training__(self):
        md = BFO.ABFOLS(root_algo_paras=self.root_algo_paras, abfols_paras=self.abfols_paras)
        self.solution, self.loss_train = md._train__()


class CsoFLNN(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, cso_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.cso_paras = cso_paras
        self.filename = "FL-CSONN-{}-nets_{}-cso_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], cso_paras)

    def _training__(self):
        md = CSO.BaseCSO(root_algo_paras=self.root_algo_paras, cso_paras=self.cso_paras)
        self.solution, self.loss_train = md._train__()


class AbcFlnn(RootHybridFlnn):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, abc_paras=None):
        RootHybridFlnn.__init__(self, root_base_paras, root_hybrid_paras)
        self.abc_paras = abc_paras
        self.filename = "FL-ABCNN-{}-nets_{}-abc_{}".format([root_base_paras["sliding"], root_base_paras["expand_function"]],
            [root_hybrid_paras["epoch"], root_hybrid_paras["activation"], root_hybrid_paras["train_valid_rate"]], abc_paras)

    def _training__(self):
        md = ABC.BaseABC(root_algo_paras=self.root_algo_paras, abc_paras=self.abc_paras)
        self.solution, self.loss_train = md._train__()


