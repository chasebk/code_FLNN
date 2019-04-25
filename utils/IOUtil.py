#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:49:35 2018
@author: thieunv
"""

import numpy as np
import pandas as pd
import csv

def save_prediction_to_csv(y_test=None, y_pred=None, filename=None, pathsave=None):
    t1 = np.concatenate((y_test, y_pred), axis=1)
    np.savetxt(pathsave + filename + ".csv", t1, delimiter=",")
    return None

def save_loss_train_to_csv(error=None, filename=None, pathsave=None):
    np.savetxt(pathsave + filename + ".csv", np.array(error), delimiter=",")
    return None

def save_all_models_to_csv(item=None, log_filename=None, pathsave=None):
    with open(pathsave + log_filename + ".csv", "a+") as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerow(item)

def save_run_test(num_run_test=None, data=None, filepath=None):
    t0 = np.reshape(data, (num_run_test, -1))
    np.savetxt(filepath, t0, delimiter=",")

def load_prediction_results(pathfile=None, delimiter=",", header=None):
    df = pd.read_csv(pathfile, sep=delimiter, header=header)
    return df.values[:, 0:1], df.values[:, 1:2]

def save_number_of_vms(data=None, pathfile=None):
    t0 = np.reshape(data, (-1, 1))
    np.savetxt(pathfile, t0, delimiter=",")

def load_number_of_vms(pathfile=None, delimiter=",", header=None):
    df = pd.read_csv(pathfile, sep=delimiter, header=header)
    return df.values[:, 0:1]



def save_scaling_results_to_csv(data=None, path_file=None):
    np.savetxt(path_file + ".csv", np.array(data), delimiter=",")
    return None

def read_dataset_file(filepath=None, usecols=None, header=0, index_col=False, inplace=True):
    df = pd.read_csv(filepath, usecols=usecols, header=header, index_col=index_col)
    df.dropna(inplace=inplace)
    return df.values

def save_formatted_data_csv(dataset=None, filename=None, pathsave=None):
    np.savetxt(pathsave + filename + ".csv", dataset, delimiter=",")
    return None