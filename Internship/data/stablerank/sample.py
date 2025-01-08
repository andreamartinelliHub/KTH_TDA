#!/usr/bin/env python3
"""
Created on Sun June 13 18:27 2021

@author: wojtek
"""

import numpy as np


class Sample(object):
    """
    Objects in this class represent samples of various objects.

    Attributes
    ----------
    sample: str, ndarray
        can be either:
            - string "all", or
            - string "empty", or
            - 2D ndarray of integers.
    sample_size: int
        integer which is:
            - the integer describing the size of the sampled object, in case the Sample attribute is
            the string "all", or
            - the number of columns of the attribute sample in the case this attribute is a 2D ndarray.
    number_instances: int
        integer which is:
            - 1 if the Sample attribute is the string "all",
            - the number of rows of the attribute sample in the case this attribute is a 2D ndarray.
    """
    def __init__(self, s, sample_size=None):
        """
        Parameters
        ----------
        s: str, list[list[int]], list[list[Real]], list[int], list[Real], ndarray
            can be either:
            - string "all", or
            - string "empty", or
            - 2D array like of integers, or
            - 1D array like of integers.
        sample_size: None, int
            either:
            - None, or
            - integer, describing the Sample size. This parameter is only relevant in the case the parameter s is
            the string "all".
        """
        if isinstance(s, str) and s == "all" and isinstance(sample_size, int):
            self.sample = "all"
            self.number_instances = 1
            self.sample_size = sample_size
        elif isinstance(s, str) and s == "empty":
            self.sample = "empty"
            self.number_instances = 0
            self.sample_size = 0
        else:
            ss = np.asarray(s, dtype=int)
            if np.ndim(ss) == 2:
                if np.shape(ss)[0] == 0 or np.shape(ss)[1] == 0:
                    self.sample = "empty"
                    self.number_instances = 0
                    self.sample_size = 0
                else:
                    self.sample = ss
                    self.sample_size = ss.shape[1]
                    self.number_instances = ss.shape[0]
            elif np.ndim(ss) == 1:
                if len(ss) == 0:
                    self.sample = "empty"
                    self.number_instances = 0
                    self.sample_size = 0
                else:
                    self.sample = np.array([ss])
                    self.sample_size = len(ss)
                    self.number_instances = 1
            else:
                raise ValueError("""the parameter Sample should be either string "all", or string "empty", 
                or 1D or 2D array like, in the case the parameter Sample is the string "all", then the parameter 
                sample_size needs to be specified (as an integer)""")


def get_sample(number_instances, sample_size, probabilities):
    """
    Produces a Sample object representing a sampling with respect to the Distribution specified by the parameter
    probabilities.

    Parameters
    ----------
    number_instances: int
        positive integer encoding the number of times a sampling is performed.
    sample_size: int
        positive integer specifying the Sample size.
    probabilities: list, list[Real], ndarray, boolean
        can be either:
        - a list ["uniform", size] where size is a positive integer, or
        - 1D array like object of non negative real numbers, or
        - boolean False

    Returns
    ----------
    Sample object
    """
    p = probabilities
    if isinstance(p, bool):
        return Sample("empty")
    elif isinstance(p, (tuple, list)) and isinstance(p[0], str) and p[0] == "uniform":
        if sample_size <= p[1]:
            out = np.zeros([number_instances, sample_size], dtype=int)
            for inst in range(number_instances):
                out[inst] = np.sort(np.random.choice(p[1], sample_size, replace=False))
            return Sample(out)
        else:
            return Sample("empty")
    else:
        size = np.count_nonzero(p)
        if sample_size <= size:
            out = np.zeros([number_instances, sample_size], dtype=int)
            for inst in range(number_instances):
                out[inst] = np.sort(np.random.choice(len(p), sample_size, replace=False, p=p))
            return Sample(out)
        else:
            return Sample("empty")
