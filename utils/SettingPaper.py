###### Config for test

###: Variables
ggtrace_cpu = [
    "data/formatted/",      # pd.readfilepath
    [1],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "cpu/",                 # path_save_result
]

ggtrace_ram = [
    "data/formatted/",      # pd.readfilepath
    [2],                    # usecols trong pd
    False,                  # multi_output
    None,                   # output_idx
    "ram/",                 # path_save_result
]

ggtrace_multi_cpu = [
    "data/formatted/",      # pd.readfilepath
    [1, 2],                 # usecols trong pd
    False,                  # multi_output
    0,                      # output_idx
    "multi_cpu/",           # path_save_result
]

ggtrace_multi_ram = [
    "data/formatted/",      # pd.readfilepath
    [1, 2],                 # usecols trong pd
    False,                  # multi_output
    1,                      # output_idx
    "multi_ram/",           # path_save_result
]

giang1 = [
    "data/formatted/giang/",      # pd.readfilepath
    [3],                 # usecols trong pd
    False,                  # multi_output
    None,                      # output_idx
    "giang1/",           # path_save_result
]


######################## Paras according to the paper

####: FLNN
flnn_paras = {
    "sliding_window": [2, 3, 4, 5],
    "expand_function": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu", "tanh"],

    "epoch": [1500],
    "learning_rate": [0.025],                # 100 -> 900
    "batch_size": [64],                                 # 0.85 -> 0.97
    "beta": [0.90]                                      # 0.005 -> 0.10
}

####: MLNN-1HL
mlnn1hl_paras_final = {
    "sliding_window": [2, 5, 10],
    "expand_function": [None],
    "hidden_sizes" : [[5] ],
    "activations": [("elu", "elu")],    # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.01],
    "epoch": [1000],
    "batch_size": [128],
    "optimizer": ["adam"],          # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}

####: MLNN-1HL
mlnn2hl_paras_final = {
    "sliding_window": [2, 5, 10],
    "expand_function": [None],
    "hidden_sizes" : [[5, 3] ],
    "activations": [("elu", "elu", "elu")],  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "learning_rate": [0.0001],
    "epoch": [2000],
    "batch_size": [128],
    "optimizer": ["adam"],              # GradientDescentOptimizer, AdamOptimizer, AdagradOptimizer, AdadeltaOptimizer
    "loss": ["mse"]
}





# ========================== Hybrid FLNN =================================

#### : FL-GANN
flgann_giang1_paras = {
    "sliding_window": [2, 3, 5],
    "expand_function": [0, 1, 2, 3, 4],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu", "tanh"],                  # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [700],
    "pop_size": [250],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}


#### : FL-GANN
flgann_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu", "tanh"],      # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [10],
    "pop_size": [200],                  # 100 -> 900
    "pc": [0.95],                       # 0.85 -> 0.97
    "pm": [0.025],                      # 0.005 -> 0.10
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : DE-MLNN
fldenn_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu", "tanh"],      # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [20],
    "pop_size": [200],                  # 10 * problem_size
    "wf": [0.8],                        # Weighting factor
    "cr": [0.9],                        # Crossover rate
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : PSO-FLNN
flpsonn_paras = {
    "sliding_window": [2, 5, 10],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu", "tanh"],      # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [50],
    "pop_size": [200],                  # 100 -> 900
    "w_minmax": [(0.4, 0.9)],  # [0-1] -> [0.4-0.9]      Trong luong cua con chim
    "c_minmax": [(1.2, 1.2)],  # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]     # [0-2]   Muc do anh huong cua local va global
    # r1, r2 : random theo tung vong lap
    # delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : BFO-FLNN
flbfonn_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu"],      # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "pop_size": [50],
    "Ci": [0.01],                       # step_size
    "Ped": [0.25],                      # p_eliminate
    "Ns": [2],                          # swim_length
    "Ned": [6],                                 #  elim_disp_steps
    "Nre": [2],                                 # repro_steps
    "Nc": [30],                                 # chem_steps
    "attract_repel": [(0.1, 0.2, 0.1, 10)],    # [ d_attr, w_attr, h_repel, w_repel ]

    "domain_range": [(-1, 1)]           # lower and upper bound
}

#### : ABFOLS-FLNN
flabfolsnn_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu"],              # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [100],
    "pop_size": [200],               # 100 -> 900
    "Ci": [(0.1, 0.00001)],         # C_s (start), C_e (end)  -=> step size # step size in BFO
    "Ped": [0.25],                  # p_eliminate
    "Ns": [4],                      # swim_length
    "N_minmax": [(3, 40)],          # (Dead threshold value, split threshold value) -> N_adapt, N_split

    "domain_range": [(-1, 1)]  # lower and upper bound
}

#### : CSO-FLNN
flcsonn_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu"],              # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [100],
    "pop_size": [200],                  # 100 -> 900
    "mixture_ratio": [0.15],            #
    "smp": [10],                        # seeking memory pool, 10 clones  (greater is better but more need time training)
    "spc": [True],                      # self-position considering
    "cdc": [0.8],               # counts of dimension to change  (greater is better)
    "srd": [0.01],              # seeking range of the selected dimension (lower is better but slow searching domain)
    "w_minmax": [(0.4, 0.9)],   # same in PSO
    "c1": [0.4],                # same in PSO
    "selected_strategy": [0],   # 0: best fitness, 1: tournament, 2: roulette wheel, 3: random  (decrease by quality)

    "domain_range": [(-1, 1)]  # lower and upper bound
}

#### : ABC-FLNN
flabcnn_paras = {
    "sliding_window": [2],
    "expand_function": [0],             # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": ["elu"],              # 0: elu, 1:relu, 2:tanh, 3:sigmoid
    "train_valid_rate": [(0.6, 0.4)],

    "epoch": [100],
    "pop_size": [200],                              # 100 -> 900
    "couple_bees": [(16, 4)],     # number of bees which provided for good location and other location
    "patch_variables": [(5.0, 0.985)],  # patch_variables = patch_variables * patch_factor (0.985)
    "sites": [(3, 1)],                  # 3 bees (employeeed bees, onlookers and scouts), 1 good partition

    "domain_range": [(-1, 1)]  # lower and upper bound
}






