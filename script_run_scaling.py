from scaling.ProactiveSLAScaling import SLABasedOnVms as BrokerScaling
from utils.IOUtil import load_number_of_vms, save_scaling_results_to_csv
from utils.MetricUtil import ADI
import numpy as np

model_names = {"ann": "ann", "mlnn": "mlnn", "flnn": "flnn", "fl_gann": "fl_gann", "fl_bfonn": "fl_bfonn"}
input_types = {"uni": "uni", "multi": "multi"}

models = [
    {"name": model_names["ann"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_2000-lr_0.1-batch_16",
     "ram": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_1000-lr_0.05-batch_64"},
    {"name": model_names["ann"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "ANN-sliding_3-act_func1_0-act_func2_1-optimizer_0-num_hidden_20-epoch_1500-lr_0.05-batch_16",
     "ram": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_8-epoch_1000-lr_0.1-batch_64"},
    {"name": model_names["ann"],
     "sliding": 4,
     "input_type": input_types["uni"],
     "cpu": "ANN-sliding_4-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_1000-lr_0.05-batch_64",
     "ram": "ANN-sliding_4-act_func1_0-act_func2_0-optimizer_2-num_hidden_20-epoch_1000-lr_0.15-batch_64"},
    {"name": model_names["ann"],
     "sliding": 4,
     "input_type": input_types["multi"],
     "cpu": "ANN-sliding_4-act_func1_0-act_func2_0-optimizer_0-num_hidden_15-epoch_1500-lr_0.15-batch_64",
     "ram": "ANN-sliding_4-act_func1_0-act_func2_0-optimizer_2-num_hidden_20-epoch_1000-lr_0.15-batch_64"},
    {"name": model_names["ann"],
     "sliding": 5,
     "input_type": input_types["uni"],
     "cpu": "ANN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_1000-lr_0.1-batch_64",
     "ram": "ANN-sliding_5-act_func1_0-act_func2_0-optimizer_3-num_hidden_15-epoch_1000-lr_0.05-batch_256"},
    {"name": model_names["ann"],
     "sliding": 5,
     "input_type": input_types["multi"],
     "cpu": "ANN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_hidden_8-epoch_2000-lr_0.05-batch_64",
     "ram": "ANN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_hidden_20-epoch_2000-lr_0.15-batch_256"},
]
s_coffs = [1.0, 1.1]
L_adaps = [2, 3]


resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
for model in models:
    cpu_file = "results/" + model["name"] + "/cpu/" + model["cpu"] + ".csv"
    ram_file = "results/" + model["name"] + "/ram/" + model["ram"] + ".csv"

    violated_arrays = []
    adi_arrays = []
    for s_coff in s_coffs:

        violated_arr = []
        adi_arr = []
        for L_adap in L_adaps:

            broker = BrokerScaling(scaling_coefficient=s_coff, adaptation_len=L_adap)
            neural_net = broker.sla_violate(cpu_file, ram_file)
            eval_scaler = ADI(metric='CPU Utilisation %')
            adi = sum(np.array(eval_scaler.calculate_ADI(resource_used=resource_real_used, resource_allocated=neural_net[1][-1])))

            violated_arr.append(neural_net[0])
            adi_arr.append(adi)

        violated_arrays.append(violated_arr)
        adi_arrays.append(adi_arr)

    violated_path_file = "results/scaling/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_violated"
    adi_path_file = "results/scaling/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_adi"

    save_scaling_results_to_csv(violated_arrays, violated_path_file)
    save_scaling_results_to_csv(adi_arrays, adi_path_file)




