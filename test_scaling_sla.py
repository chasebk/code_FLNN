from model.scaling.ProactiveSLAScaling import SLABasedOnVms as BrokerScaling
from utils.IOUtil import load_prediction_results, load_number_of_vms
from utils.MetricUtil import ADI
import numpy as np

def sla_violate(cpu, ram, broker):
    cpu_actual, cpu_predict = load_prediction_results(cpu)
    ram_actual, ram_predict = load_prediction_results(ram)

    resources_actuals = np.concatenate( (cpu_actual, ram_actual), axis= 1 )
    resources_predicts = np.concatenate((cpu_predict, ram_predict), axis=1)

    number_of_VMs = broker.allocate_VMs(resources_actuals=resources_actuals, resources_predicts=resources_predicts)

    cpu_alloc = number_of_VMs * broker.capacity_VM[0]
    ram_alloc = number_of_VMs * broker.capacity_VM[1]

    c = np.array((cpu_actual >= cpu_alloc))
    d = np.array((ram_actual >= ram_alloc))
    z = np.concatenate((c, d), axis=1)
    e = np.array([(x or y) for x, y in z])
    return float(len(e[e == True])) * 100 / len(e), (cpu_alloc, ram_alloc, number_of_VMs)

broker = BrokerScaling(scaling_coefficient=1.1, adaptation_len=10)


ann_cpu_file = "results/ann/cpu/ANN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_1000-lr_0.05-batch_256.csv"
ann_ram_file = "results/ann/ram/ANN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_1000-lr_0.05-batch_256.csv"
ann = sla_violate(ann_cpu_file, ann_ram_file, broker)
print(ann[0])
resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
eval_scaler = ADI(metric='CPU Utilisation %')
time_util_ann = np.array(eval_scaler.calculate_ADI(resource_used=resource_real_used, resource_allocated=ann[1][-1]))
print("ADI ANN: %s"%sum(time_util_ann))



flnn_cpu_file = "results/flnn/cpu/flnn-sliding_2-ex_func_2-act_func_1-epoch_2000-lr_0.15-batch_256-beta_0.9.csv"
flnn_ram_file = "results/flnn/ram/flnn-sliding_2-ex_func_2-act_func_1-epoch_2000-lr_0.15-batch_256-beta_0.9.csv"
flnn = sla_violate(flnn_cpu_file, flnn_ram_file, broker)
print(flnn[0])
resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
eval_scaler = ADI(metric='CPU Utilisation %')
time_util_flnn = np.array(eval_scaler.calculate_ADI(resource_used=resource_real_used, resource_allocated=flnn[1][-1]))
print("ADI FLNN: %s"%sum(time_util_flnn))




range_plot = (0, 1000)
real_used = resource_real_used[range_plot[0]:range_plot[-1]].tolist()
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xlabel('Time')
ax.set_ylabel('Number of VMs')
ax.set_title('Scaling VMs by One Dimension Input')
ann_s = ann[1][-1][range_plot[0]:range_plot[-1]]
flnn_s = flnn[1][-1][range_plot[0]:range_plot[-1]]

# ax.set_color_cycle(['red','blue','green','cyan','orange'])
ax.plot(real_used, label='Real Usage')
ax.plot(ann_s,'--',label='ANN')
ax.plot(flnn_s,'--',label='FLNN')

ax.legend(loc='upper right')
plt.tight_layout()
plt.show()


















