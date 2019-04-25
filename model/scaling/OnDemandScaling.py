from model.scaling.BaseStrategy import BaseStrategy
import numpy as np

class OnDemandScaling(BaseStrategy):
    def __init__(self, capacity_VM = (0.25, 0.03), metrics=('CPU','RAM')):
        self.capacity_VM = np.array(capacity_VM)
        self.metrics = metrics
    """
        Allocate VMs based on resource metric usage
    """

    def __allocate_VM(self, resource, idx):
        """
        :param resources_usage: numpy array
        :return:
        """
        capa = self.capacity_VM[idx]
        return np.ceil(resource / capa)

    def __allocate_VMs(self, resources):
        return np.ceil(resources / self.capacity_VM)

    def allocate_VMs(self, resources_usage=None):
        number_of_VMs = self.__allocate_VMs(resources_usage)
        return np.reshape( np.max(number_of_VMs, axis=1), (-1, 1) )

    def allocate_VMs_by_idx(self, resources_usage=None):
        if resources_usage is not np.array :
            resources_usage = np.array(resources_usage)
        number_of_VMs = []
        for idx in range(len(self.metrics)):
            number_of_VMs.append( self.__allocate_VM(resources_usage[:, idx], idx) )
        return np.reshape( np.max(number_of_VMs, axis=1), (-1, 1) )

