from model.scaling.ProactiveSLAScaling import SLABasedOnVms as BrokerScaling
from utils.IOUtil import load_number_of_vms, save_scaling_results_to_csv
import numpy as np

model_names = {"fl_bfonn": "fl_bfonn"}
input_types = {"uni": "uni", "multi": "multi"}

models = [
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


s_coffs = [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5 ]
L_adaps = [ 5 ]


resource_real_used = load_number_of_vms('vms_real_used_CPU_RAM.csv')
for model in models:

    if model["input_type"] == "multi":
        cpu_file = "results/" + model["name"] + "/multi_cpu/" + model["cpu"] + ".csv"
        ram_file = "results/" + model["name"] + "/multi_ram/" + model["ram"] + ".csv"
    else:
        cpu_file = "results/" + model["name"] + "/cpu/" + model["cpu"] + ".csv"
        ram_file = "results/" + model["name"] + "/ram/" + model["ram"] + ".csv"

    for s_coff in s_coffs:

        for L_adap in L_adaps:
            broker = BrokerScaling(scaling_coefficient=s_coff, adaptation_len=L_adap)
            vm_predicted, vm_actual, vm_allocated, sla = broker.get_predicted_and_allocated_vms(cpu_file, ram_file)

            vms_arr = np.concatenate((vm_predicted, vm_allocated, vm_actual), axis=1)

            filepathresults = "results/scaling3/sliding" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_vms-s_" + str(s_coff) + "-L_" + str(L_adap)
            filepathsla = "results/scaling3/sliding" + str(model["sliding"]) + "/" + model["input_type"] + "/" + model["name"] + "_SLA-s_" + str(s_coff) + "-L_" + str(L_adap)
            save_scaling_results_to_csv(vms_arr, filepathresults)
            save_scaling_results_to_csv(sla, filepathsla)
            del vms_arr
            del broker



