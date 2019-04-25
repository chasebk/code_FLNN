from sklearn.model_selection import ParameterGrid
from model.main.traditional_ffnn import Mlnn1HL
from utils.IOUtil import read_dataset_file
from utils.SettingPaper import mlnn1hl_paras_final as param_grid
from utils.SettingPaper import ggtrace_cpu, ggtrace_ram, ggtrace_multi_cpu, ggtrace_multi_ram

rv_data = [ggtrace_cpu, ggtrace_ram, ggtrace_multi_cpu, ggtrace_multi_ram]
data_file = ["google_5m", "google_5m", "google_5m", "google_5m"]
test_type = "normal"                ### normal: for normal test, stability: for n_times test
run_times = None

if test_type == "normal":           ### For normal test
    run_times = 1
    pathsave = "paper/results/test/"
    all_model_file_name = "log_models"
elif test_type == "stability":      ### For stability test (n times run with the same parameters)
    run_times = 15
    pathsave = "paper/results/stability/"
    all_model_file_name = "stability_mlnn"
else:
    pass

def train_model(item):
    root_base_paras = {
        "dataset": dataset,
        "data_idx": (0.8, 0, 0.2),
        "sliding": item["sliding_window"],
        "expand_function": item["expand_function"],
        "multi_output": requirement_variables[2],
        "output_idx": requirement_variables[3],
        "method_statistic": 0,                  # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
        "log_filename": all_model_file_name,
        "path_save_result": pathsave + requirement_variables[4],
        "test_type": test_type,
        "draw": True,
        "print_train": 2                        # 0: nothing, 1: full details, 2: short
    }
    root_mlnn1hl_paras = {
        "hidden_sizes": item["hidden_sizes"], "epoch": item["epoch"], "batch_size": item["batch_size"],
        "lr": item["learning_rate"], "activations": item["activations"], "optimizer": item["optimizer"], "loss": item["loss"]
    }
    md = Mlnn1HL(root_base_paras=root_base_paras, root_mlnn1hl_paras=root_mlnn1hl_paras)
    md._running__()

for _ in range(run_times):
    for loop in range(len(rv_data)):
        requirement_variables = rv_data[loop]
        filename = requirement_variables[0] + data_file[loop] + ".csv"
        dataset = read_dataset_file(filename, requirement_variables[1])
        # Create combination of params.
        for item in list(ParameterGrid(param_grid)):
            train_model(item)
