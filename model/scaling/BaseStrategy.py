class BaseStrategy(object):
    def basic_allocate_VMs(self, resource_used):
        pass

    def sla_violation(self, actual_used=None, allocated = None):
        delta = actual_used - allocated
        violation_count = len(delta[delta>0])
        return float(violation_count) / len(actual_used)

