from model.scaling.ProactiveSLAScaling import SLABasedOnVms as BrokerScaling
from utils.IOUtil import load_number_of_vms, save_scaling_results_to_csv
from utils.MetricUtil import ADI
import numpy as np

model_names = {"ann": "ann", "mlnn":"mlnn", "flnn": "flnn", "fl_gann": "fl_gann", "fl_bfonn": "fl_bfonn", "fl_abfolsnn": "fl_abfolsnn"}
input_types = {"uni": "uni", "multi": "multi"}

models = [
{"name": model_names["ann"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_12-epoch_2000-lr_0.1-batch_64",
     "ram": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_12-epoch_2000-lr_0.1-batch_64"},
    {"name": model_names["ann"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_12-epoch_2000-lr_0.1-batch_64",
     "ram": "ANN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_hidden_12-epoch_2000-lr_0.1-batch_64"},

{"name": model_names["mlnn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_12_h2_20_h3_5-epoch_2000-lr_0.1-batch_64",
     "ram": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_12_h2_20_h3_5-epoch_2000-lr_0.1-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_12_h2_20_h3_5-epoch_2000-lr_0.1-batch_64",
     "ram": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_12_h2_20_h3_5-epoch_2000-lr_0.1-batch_64"},

{"name": model_names["flnn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "flnn-sliding_3-ex_func_3-act_func_1-epoch_2000-lr_0.1-batch_64-beta_0.9",
     "ram": "flnn-sliding_3-ex_func_3-act_func_1-epoch_2000-lr_0.1-batch_64-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "flnn-sliding_3-ex_func_3-act_func_1-epoch_2000-lr_0.1-batch_64-beta_0.9",
     "ram": "flnn-sliding_3-ex_func_3-act_func_1-epoch_2000-lr_0.1-batch_64-beta_0.9"},

{"name": model_names["fl_gann"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "FL_GANN-sliding_3-ex_func_0-act_func_1-epoch_700-pop_size_250-pc_0.95-pm_0.025",
     "ram": "FL_GANN-sliding_3-ex_func_0-act_func_1-epoch_700-pop_size_250-pc_0.95-pm_0.025"},
    {"name": model_names["fl_gann"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "FL_GANN-sliding_3-ex_func_0-act_func_1-epoch_700-pop_size_250-pc_0.95-pm_0.025",
     "ram": "FL_GANN-sliding_3-ex_func_0-act_func_1-epoch_700-pop_size_250-pc_0.95-pm_0.025"},

{"name": model_names["fl_bfonn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "FL_BFONN-sliding_3-ex_func_0-act_func_0-pop_size_50-steps_(4, 4, 50)-dwhw_(0.1, 0.2, 0.1, 10)-step_size_0.1-p_elim_0.25-swim_length_4",
     "ram": "FL_BFONN-sliding_3-ex_func_0-act_func_0-pop_size_50-steps_(4, 4, 50)-dwhw_(0.1, 0.2, 0.1, 10)-step_size_0.1-p_elim_0.25-swim_length_4"},
    {"name": model_names["fl_bfonn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "FL_BFONN-sliding_3-ex_func_0-act_func_0-pop_size_50-steps_(4, 4, 50)-dwhw_(0.1, 0.2, 0.1, 10)-step_size_0.1-p_elim_0.25-swim_length_4",
     "ram": "FL_BFONN-sliding_3-ex_func_0-act_func_0-pop_size_50-steps_(4, 4, 50)-dwhw_(0.1, 0.2, 0.1, 10)-step_size_0.1-p_elim_0.25-swim_length_4"},

{"name": model_names["fl_abfolsnn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "FL_ABFOLSNN-sliding_3-ex_func_0-act_func_1-epoch_1000-pop_size_50-step_size_(0.1, 1e-05)-N_split_40-N_adapt_5-p_ed_0.25-swim_length_6",
     "ram": "FL_ABFOLSNN-sliding_3-ex_func_0-act_func_1-epoch_800-pop_size_50-step_size_(0.1, 1e-05)-N_split_40-N_adapt_3-p_ed_0.25-swim_length_4"},
    {"name": model_names["fl_abfolsnn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "FL_ABFOLSNN-sliding_2-ex_func_1-act_func_1-epoch_1000-pop_size_50-step_size_(0.1, 1e-05)-N_split_60-N_adapt_3-p_ed_0.25-swim_length_4",
     "ram": "FL_ABFOLSNN-sliding_2-ex_func_2-act_func_1-epoch_800-pop_size_50-step_size_(0.1, 1e-05)-N_split_60-N_adapt_5-p_ed_0.25-swim_length_4"},
]

s_coffs = [ 1.0, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5]
L_adaps = [ 5 ]


resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
for model in models:

    if model["input_type"] == "multi":
        cpu_file = "paper/test1/multi_cpu/" + model["cpu"] + ".csv"
        ram_file = "paper/test1/multi_ram/" + model["ram"] + ".csv"
    else:
        # cpu_file = "paper/test1/cpu/" + model["cpu"] + ".csv"
        # ram_file = "paper/test1/ram/" + model["ram"] + ".csv"
        continue

    violated_arrays = []
    adi_arrays = []
    for s_coff in s_coffs:

        violated_arr = []
        adi_arr = []
        for L_adap in L_adaps:

            broker = BrokerScaling(scaling_coefficient=s_coff, adaptation_len=L_adap)
            neural_net = broker.sla_violate(cpu_file, ram_file)
            eval_scaler = ADI(lower_bound=0.6, upper_bound = 0.8, metric='CPU Utilisation %')
            adi = sum(np.array(eval_scaler.calculate_ADI(resource_used=resource_real_used, resource_allocated=neural_net[1][-1])))

            violated_arr.append(round(neural_net[0], 2))
            adi_arr.append(round(adi, 2))

        violated_arrays.append(violated_arr)
        adi_arrays.append(adi_arr)

    violated_path_file = "paper/scaling/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_violated"
    adi_path_file = "paper/scaling/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_adi"

    save_scaling_results_to_csv(violated_arrays, violated_path_file)
    save_scaling_results_to_csv(adi_arrays, adi_path_file)




