## How to read my repository
1. data: include raw and formatted data
2. envs: include conda environment and how to install conda environment 
3. utils: Helped functions such as IO, Draw, Math, Settings (for all model and parameters), Preprocessing...
4. paper: include 2 main folders: 
    * results: forecasting results of all models (3 folders inside) 
        * final: final forecasting results (runs on server)
        * test: testing forecasting results (runs on personal computer, just for test)
        * temp: nothing (just for copy of test folder)
    * scaling: scaling results
7. model: (4 folders) 
    * root: (want to understand the code, read this classes first)
        * root_base.py: root for all models (traditional, hybrid and variants...) 
        * root_algo.py: root for all optimization algorithms
        * traditional: root for all traditional models (inherit: root_base)
        * hybrid: root for all hybrid models (inherit: root_base)
    * optimizer: (this classes inherit: root_algo.py)
        * evolutionary: include algorithms related to evolution algorithm such as GA, DE,..
        * swarm: include algorithms related to swarm optimization such as PSO, CSO, BFO, ...
    * main: (final models)
        * this classes will use those optimizer above and those root (traditional, hybrid) above 
        * the running files (outside with the orginial folder: prediction_flnn) will call this classes
        * the traditional models will use single file such as: traditional_ffnn, traditional_flnn,...
        * the hybrid models will use 2 files, example: hybrid_flnn.py and GA.py (optimizer files)
8. special files
    * vms_real_used_CPU_RAM.csv (the real amount of CPU and RAM used in cloud): calculated by get_real_Vms_usages.py file
    * *_scipt.py: running files (*: represent model) such as flgann_script.py => FLNN + GA
    
## Notes
1. To improve the speed of Pycharm when opening (because Pycharm will indexing when opening), you should right click to 
paper and data folder => Mark Directory As  => Excluded

2. When runs models, you should copy the running files to the original folder (prediction_flnn folder)

3. Make sure you active the environment before run the running files 
* For terminal on linux
```code
    source activate environment_name 
    python running_file.py 
```
4. In paper/results/final model includes folder's name represent the data such as 
```code
cpu: input model would be cpu, output model would be cpu 
ram: same as cpu
multi_cpu : input model would be cpu and ram, output model would be cpu 
multi_ram : input model would be cpu and ram, output model would be ram
multi : input model would be cpu and ram, output model would be cpu and ram
```

## Model
1. ANN (1 HL) => mlnn1hl_script.py
2. FLNN => flnn_script.py
3. FL-GANN => flgann_script.py
4. FL-DENN => fldenn_script.py
5. FL-PSONN => flpsonn_script.py
6. FL-ABCNN => flabcnn_script.py
7. FL-BFONN => flbfonn_script.py
8. FL-ABFOLSNN => flabfonn_script.py
9. FL-CSONN => flcsonn_script.py

## Publications
* If you see my code and data useful and use it, please cites us here

    * Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

    * Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.

* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com

* Take a look at this repos, the simplify code using python (numpy) for all algorithms above. (without neural networks)
	
	* https://github.com/thieunguyen5991/metaheuristics

