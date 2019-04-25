from model.scaling.OnDemandScaling import OnDemandScaling as BrokerScaling
from utils.IOUtil import load_prediction_results
from utils.IOUtil import save_number_of_vms
import numpy as np

def get_real_vms_usages(cpu, ram, broker):
    cpu_actual, cpu_predict = load_prediction_results(cpu)
    ram_actual, ram_predict = load_prediction_results(ram)

    resources_actual = np.concatenate( (cpu_actual, ram_actual), axis= 1 )
    number_of_VMs = np.array(broker.allocate_VMs(resources_usage=resources_actual))
    save_number_of_vms(number_of_VMs, "vms_real_used_CPU_RAM.csv")

broker = BrokerScaling()

cpu_file = "results/ann/cpu/ANN-sliding_2-act_func1_0-act_func2_0-optimizer_0-num_hidden_8-epoch_1000-lr_0.1-batch_16.csv"
ram_file = "results/ann/ram/ANN-sliding_2-act_func1_0-act_func2_0-optimizer_0-num_hidden_8-epoch_1000-lr_0.1-batch_16.csv"
get_real_vms_usages(cpu_file, ram_file, broker)



















