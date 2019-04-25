#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:13:13 2018
@author: thieunv

Link : http://sci-hub.tw/10.1109/iccat.2013.6521977
https://en.wikipedia.org/wiki/Laguerre_polynomials

"""

import numpy as np

def itself(x):
    return x
def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)
def relu(x):
    return np.maximum(0, x)
def tanh(x):
    return np.tanh(x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_self(x):
    return 1
def derivative_elu(x, alpha=1):
    return np.where(x < 0, x + alpha, 1)
def derivative_relu(x):
    return np.where(x < 0, 0, 1)
def derivative_tanh(x):
    return 1 - np.power(x, 2)
def derivative_sigmoid(x):
    return np.multiply(x, 1-x)


def expand_chebyshev(x):
    x1 = x
    x2 = 2 * np.power(x, 2) - 1
    x3 = 4 * np.power(x, 3) - 3 * x
    x4 = 8 * np.power(x, 4) - 8 * np.power(x, 2) + 1
    x5 = 16 * np.power(x, 5) - 20 * np.power(x, 3) + 5 * x
    return np.concatenate( (x1, x2, x3, x4, x5), axis=1 )

def expand_legendre(x):
    x1 = x
    x2 = 1/2 * ( 3 * np.power(x, 2) - 1 )
    x3 = 1/2 * (5 * np.power(x, 3) - 3 * x)
    x4 = 1/8 * ( 35 * np.power(x, 4) - 30 * np.power(x, 2) + 3 )
    x5 = 1/40 * ( 9 * np.power(x, 5) - 350 * np.power(x, 3) + 75 * x )
    return np.concatenate((x1, x2, x3, x4, x5), axis=1 )

def expand_laguerre(x):
    x1 = -x + 1
    x2 = 1/2 * ( np.power(x, 2) - 4 * x + 2)
    x3 = 1/6 * (-np.power(x, 3) + 9 * np.power(x, 2) - 18 * x + 6)
    x4 = 1/24 * (np.power(x, 4) - 16 * np.power(x, 3) + 72 * np.power(x, 2) - 96*x + 24)
    x5 = 1/120 * (-np.power(x, 5) + 25 * np.power(x, 4) - 200 * np.power(x, 3) + 600 * np.power(x, 2) - 600 * x + 120)
    return np.concatenate((x1, x2, x3, x4, x5), axis=1 )

def expand_power(x):
    x1 = x
    x2 = x1 + np.power(x, 2)
    x3 = x2 + np.power(x, 3)
    x4 = x3 + np.power(x, 4)
    x5 = x4 + np.power(x, 5)
    return np.concatenate((x1, x2, x3, x4, x5), axis=1)

def expand_trigonometric(x):
    x1 = x
    x2 = np.sin(np.pi * x) + np.cos(np.pi * x)
    x3 = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * x)
    x4 = np.sin(3 * np.pi * x) + np.cos(3 * np.pi * x)
    x5 = np.sin(4 * np.pi * x) + np.cos(4 * np.pi * x)
    return np.concatenate((x1, x2, x3, x4, x5), axis=1)