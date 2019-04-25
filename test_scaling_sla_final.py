from model.scaling.ProactiveSLAScaling import SLABasedOnVms as BrokerScaling
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



    {"name": model_names["mlnn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_10_h2_20_h3_5-epoch_1500-lr_0.1-batch_16",
     "ram": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_3-num_h1_8_h2_15_h3_5-epoch_1000-lr_0.05-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_3-num_h1_25_h2_20_h3_10-epoch_1000-lr_0.1-batch_16",
     "ram": "MLNN-sliding_3-act_func1_0-act_func2_0-optimizer_0-num_h1_10_h2_20_h3_5-epoch_1500-lr_0.05-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 4,
     "input_type": input_types["uni"],
     "cpu": "MLNN-sliding_4-act_func1_0-act_func2_0-optimizer_2-num_h1_10_h2_20_h3_5-epoch_1000-lr_0.1-batch_64",
     "ram": "MLNN-sliding_4-act_func1_0-act_func2_0-optimizer_2-num_h1_20_h2_10_h3_5-epoch_1000-lr_0.15-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 4,
     "input_type": input_types["multi"],
     "cpu": "MLNN-sliding_4-act_func1_0-act_func2_2-optimizer_0-num_h1_20_h2_10_h3_5-epoch_1500-lr_0.05-batch_16",
     "ram": "MLNN-sliding_4-act_func1_0-act_func2_2-optimizer_2-num_h1_10_h2_20_h3_5-epoch_1000-lr_0.1-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 5,
     "input_type": input_types["uni"],
     "cpu": "MLNN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_h1_20_h2_10_h3_5-epoch_1000-lr_0.15-batch_64",
     "ram": "MLNN-sliding_5-act_func1_0-act_func2_0-optimizer_3-num_h1_8_h2_15_h3_5-epoch_1500-lr_0.05-batch_64"},
    {"name": model_names["mlnn"],
     "sliding": 5,
     "input_type": input_types["multi"],
     "cpu": "MLNN-sliding_5-act_func1_0-act_func2_2-optimizer_2-num_h1_25_h2_20_h3_10-epoch_1500-lr_0.1-batch_16",
     "ram": "MLNN-sliding_5-act_func1_0-act_func2_0-optimizer_0-num_h1_8_h2_15_h3_5-epoch_1500-lr_0.1-batch_64"},



    {"name": model_names["flnn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "flnn-sliding_3-ex_func_3-act_func_0-epoch_1500-lr_0.15-batch_64-beta_0.95",
     "ram": "flnn-sliding_3-ex_func_0-act_func_0-epoch_2000-lr_0.05-batch_16-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "flnn-sliding_3-ex_func_4-act_func_0-epoch_1500-lr_0.15-batch_64-beta_0.9",
     "ram": "flnn-sliding_3-ex_func_4-act_func_0-epoch_1250-lr_0.05-batch_16-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 4,
     "input_type": input_types["uni"],
     "cpu": "flnn-sliding_4-ex_func_3-act_func_1-epoch_1250-lr_0.15-batch_16-beta_0.95",
     "ram": "flnn-sliding_4-ex_func_0-act_func_0-epoch_2000-lr_0.15-batch_16-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 4,
     "input_type": input_types["multi"],
     "cpu": "flnn-sliding_4-ex_func_4-act_func_0-epoch_2000-lr_0.1-batch_64-beta_0.95",
     "ram": "flnn-sliding_4-ex_func_2-act_func_0-epoch_2000-lr_0.15-batch_16-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 5,
     "input_type": input_types["uni"],
     "cpu": "flnn-sliding_5-ex_func_1-act_func_1-epoch_1500-lr_0.1-batch_64-beta_0.9",
     "ram": "flnn-sliding_5-ex_func_0-act_func_0-epoch_2000-lr_0.15-batch_16-beta_0.9"},
    {"name": model_names["flnn"],
     "sliding": 5,
     "input_type": input_types["multi"],
     "cpu": "flnn-sliding_5-ex_func_4-act_func_1-epoch_2000-lr_0.15-batch_16-beta_0.9",
     "ram": "flnn-sliding_5-ex_func_3-act_func_0-epoch_2000-lr_0.1-batch_16-beta_0.9"},




    {"name": model_names["fl_gann"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "FL_GANN-sliding_3-ex_func_3-act_func_0-epoch_500-pop_size_300-pc_0.9-pm_0.035",
     "ram": "FL_GANN-sliding_3-ex_func_0-act_func_1-epoch_700-pop_size_300-pc_0.95-pm_0.1"},
    {"name": model_names["fl_gann"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "FL_GANN-sliding_3-ex_func_3-act_func_0-epoch_500-pop_size_400-pc_0.9-pm_0.035",
     "ram": "FL_GANN-sliding_3-ex_func_1-act_func_0-epoch_500-pop_size_400-pc_0.95-pm_0.035"},
    {"name": model_names["fl_gann"],
     "sliding": 4,
     "input_type": input_types["uni"],
     "cpu": "FL_GANN-sliding_4-ex_func_3-act_func_0-epoch_500-pop_size_400-pc_0.95-pm_0.1",
     "ram": "FL_GANN-sliding_4-ex_func_3-act_func_3-epoch_500-pop_size_300-pc_0.9-pm_0.035"},
    {"name": model_names["fl_gann"],
     "sliding": 4,
     "input_type": input_types["multi"],
     "cpu": "FL_GANN-sliding_4-ex_func_1-act_func_3-epoch_700-pop_size_300-pc_0.9-pm_0.035",
     "ram": "FL_GANN-sliding_4-ex_func_1-act_func_3-epoch_500-pop_size_200-pc_0.9-pm_0.035"},
    {"name": model_names["fl_gann"],
     "sliding": 5,
     "input_type": input_types["uni"],
     "cpu": "FL_GANN-sliding_5-ex_func_3-act_func_0-epoch_700-pop_size_200-pc_0.9-pm_0.05",
     "ram": "FL_GANN-sliding_5-ex_func_3-act_func_3-epoch_500-pop_size_400-pc_0.9-pm_0.05"},
    {"name": model_names["fl_gann"],
     "sliding": 5,
     "input_type": input_types["multi"],
     "cpu": "FL_GANN-sliding_5-ex_func_3-act_func_3-epoch_700-pop_size_400-pc_0.95-pm_0.05",
     "ram": "FL_GANN-sliding_5-ex_func_3-act_func_3-epoch_500-pop_size_300-pc_0.9-pm_0.05"},



    {"name": model_names["fl_bfonn"],
     "sliding": 3,
     "input_type": input_types["uni"],
     "cpu": "FL_BFONN-sliding_3-ex_func_3-act_func_0-pop_size_70-elim_disp_steps_2-repro_steps_5-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_4",
     "ram": "FL_BFONN-sliding_3-ex_func_2-act_func_0-pop_size_100-elim_disp_steps_1-repro_steps_5-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_8"},
    {"name": model_names["fl_bfonn"],
     "sliding": 3,
     "input_type": input_types["multi"],
     "cpu": "FL_BFONN-sliding_3-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_2-repro_steps_5-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_8",
     "ram": "FL_BFONN-sliding_3-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_1-repro_steps_3-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.1-p_eliminate_0.25-swim_length_8"},
    {"name": model_names["fl_bfonn"],
     "sliding": 4,
     "input_type": input_types["uni"],
     "cpu": "FL_BFONN-sliding_4-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_2-repro_steps_3-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.1-p_eliminate_0.25-swim_length_8",
     "ram": "FL_BFONN-sliding_4-ex_func_2-act_func_0-pop_size_100-elim_disp_steps_2-repro_steps_5-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.15-p_eliminate_0.25-swim_length_4"},
    {"name": model_names["fl_bfonn"],
     "sliding": 4,
     "input_type": input_types["multi"],
     "cpu": "FL_BFONN-sliding_4-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_2-repro_steps_5-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_8",
     "ram": "FL_BFONN-sliding_4-ex_func_3-act_func_0-pop_size_70-elim_disp_steps_1-repro_steps_5-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_4"},
    {"name": model_names["fl_bfonn"],
     "sliding": 5,
     "input_type": input_types["uni"],
     "cpu": "FL_BFONN-sliding_5-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_1-repro_steps_3-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_4",
     "ram": "FL_BFONN-sliding_5-ex_func_3-act_func_0-pop_size_50-elim_disp_steps_2-repro_steps_5-chem_steps_80-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.1-p_eliminate_0.25-swim_length_4"},
    {"name": model_names["fl_bfonn"],
     "sliding": 5,
     "input_type": input_types["multi"],
     "cpu": "FL_BFONN-sliding_5-ex_func_3-act_func_0-pop_size_100-elim_disp_steps_2-repro_steps_5-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_8",
     "ram": "FL_BFONN-sliding_2-ex_func_3-act_func_0-pop_size_50-elim_disp_steps_2-repro_steps_3-chem_steps_60-d_attr_0.1_w_attr_0.2-h_rep_0.1_w_rep_10-step_size_0.05-p_eliminate_0.25-swim_length_4"},
]

s_coffs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
L_adaps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
for model in models:

    if model["input_type"] == "multi":
        cpu_file = "results/" + model["name"] + "/multi_cpu/" + model["cpu"] + ".csv"
        ram_file = "results/" + model["name"] + "/multi_ram/" + model["ram"] + ".csv"
    else:
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
            eval_scaler = ADI(lower_bound=0.6, upper_bound = 0.8, metric='CPU Utilisation %')
            adi = sum(np.array(eval_scaler.calculate_ADI(resource_used=resource_real_used, resource_allocated=neural_net[1][-1])))

            violated_arr.append(round(neural_net[0], 2))
            adi_arr.append(round(adi, 2))

        violated_arrays.append(violated_arr)
        adi_arrays.append(adi_arr)

    violated_path_file = "results/scaling1/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_violated"
    adi_path_file = "results/scaling1/sliding_" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_adi"

    save_scaling_results_to_csv(violated_arrays, violated_path_file)
    save_scaling_results_to_csv(adi_arrays, adi_path_file)




