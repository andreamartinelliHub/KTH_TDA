#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:30:38 2021

@author: wojtek
"""

import numpy as np

import pandas as pd


def randomchoice(sample_size, p):
    """
    Parameters
    ----------
    sample_size: int
        positive integer which describes the size of the chosen Sample.
    p: boolean, list, list[Real], ndarray
        either:
        - boolean False, or
        - a tuple ["uniform", size] where size is an integer. In this case the parameter is converted into a
        uniform probability Distribution on the list <code>np.arange(size)</code>
        - 1D array like of real numbers that sum up to 1.

    Returns
    -------
        - string "empty" in the case the parameter p is boolean False, or
        <code>sample_size > np.count_nonzero(P)</code> in the case P is 1D array like, or
        - ndarray of integers of size sample_size.
    """
    if isinstance(p, bool):
        return "empty"
    elif isinstance(p, (tuple, list)) and p[0] == "uniform" and sample_size <= p[1]:
        return np.sort(np.random.choice(p[1], sample_size, replace=False))
    elif sample_size <= np.count_nonzero(p):
        return np.sort(np.random.choice(len(p), sample_size, replace=False, p=p/np.sum(p)))
    else:
        return "empty"


def build_pd_Series(parameters):
    extended = []
    single = []
    I = []

    for p in parameters.keys():
        if isinstance(parameters[p], dict):
            extended.append(p)
            I.append(list(parameters[p].keys()))
        elif isinstance(parameters[p], pd.core.series.Series):
            extended.append(p)
            I.append(parameters[p].index.tolist())
        else:
            single.append(p)  
    if len(extended) == 0:
        return pd.Series(parameters)
    elif len(extended) == 1:
        index = pd.Index(I[0], name = extended[0])
        outcome = pd. DataFrame(index = index, 
                                columns = parameters.keys(),
                                dtype = object)
        for i in index:
            if isinstance(parameters[extended[0]], dict): 
                outcome.at[i,extended[0]] = parameters[extended[0]][i]
            else:
                outcome.at[i,extended[0]] = parameters[extended[0]].at[i]
            for n in single:
                outcome.at[i,n] = parameters[n]
        return outcome
    else: 
        index = pd.MultiIndex.from_product(I, names = extended)
        outcome = pd. DataFrame(index = index, 
                                columns = parameters.keys(),
                                dtype = object)
        for i in index:
            j = 0
            while j < len(extended):
                if isinstance(parameters[extended[j]], dict): 
                    outcome.at[i,extended[j]] = parameters[extended[j]][i[j]]
                else:
                    outcome.at[i,extended[j]] = parameters[extended[j]].at[i[j]]
                j = j+1
            for n in single:
                outcome.at[i,n] = parameters[n]
                
        return outcome




    

            
  
# P = {"a":{1:2, 2: "w"}, "b":pd.Series({"w":1,"v":3}), "c":2}

# P1 = {"a":{1:2, 2: "w"}, "b":2}


# P2 = {"b":pd.Series({"w":1}), "c":2}

# P3 = {"a":1, "b": "w"}

# B = build_pd_Series(P2)



# print(B)

# # # for i in B.index:
# # #     print(i[1])
    