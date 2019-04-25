#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:05:56 2018
@author: thieunv
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def draw_predict(fig_id=None, y_test=None, y_pred=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('CPU')
    plt.xlabel('Timestamp')
    plt.legend(['Actual', 'Predict'], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None

def draw_predict_with_error(fig_id=None, data=None, error=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(data[0])
    plt.plot(data[1])
    plt.ylabel('Real value')
    plt.xlabel('Point')
    plt.legend(['Predict y... RMSE= ' + str(error[0]), 'Test y... MAE= ' + str(error[1])], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None
