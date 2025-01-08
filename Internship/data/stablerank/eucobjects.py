#!/usr/bin/env python3
"""
Created on Sun June 21 13:57 2021

@author: wojtek
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial as spatial
from ripser import ripser

from stablerank.rtorf import Pcf
from stablerank.rtorf import Density
from stablerank.rtorf import Pcnif
from stablerank.rtorf import Distribution
from stablerank.rtorf import Contour

from stablerank.barcodes import BC
from stablerank.barcodes import empty_bc
from stablerank.barcodes import one_point_bc

from stablerank.distances import Distance
from stablerank.distances import SDistance

from stablerank.sample import Sample
from stablerank.sample import get_sample

from stablerank.covers import Cover


inf = float("inf")


class EucObject(object):
    """
    Objects in this class represent finite subsets of $R^n$ referred to also as point clouds. To define an EucObject
    we need to specify a 2D array like object of real numbers. Points of the EucObject are represented
    by the rows of this array.

    Attributes
    ----------
    points: ndarray
        2D ndarray of real numbers whose rows represent points of the EucObject.
    size: int
        integer describing number of points in the EucObject. It is the number of rows in the attribute points
    dim: int
        integer describing the dimension of the space in which the EucObject is embedded. It is the number of columns
        in the attribute points.
    """
    def __init__(self, points):
        """
         Parameters
         ----------
         points: list[list[Real]], ndarray
            2D array like of real numbers whose rows represent points of the EucObject.
        """
        self.points = np.asarray(points, dtype=float)
        self.size = len(self.points)
        self.dim = self.points.shape[1]

    def plot(self, color=None, s=None):
        """
        In case the dimension of the object is 2 or 3, a pyplot plot is produced.

        Parameters
        ----------
        color: None, str
            optional, default None, either None, or any expression describing the color in pyplot plot.
        s: None, Real
            optional, default None, either None, or a positive real n umber describing the size of the dots used in
            the plot.

        Returns
        ----------
        pyplot plot.
        """
        if self.dim == 2:
            x = self.points[:, 0]
            y = self.points[:, 1]
            return plt.scatter(x, y, s=s, c=color)
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            x = self.points[:, 0]
            y = self.points[:, 1]
            z = self.points[:, 2]
            return ax.scatter(x, y, z, s=s, c=color)
        else:
            raise ValueError("We can only plot 2 and 3 d objects")

    def get_center_of_mass(self):
        """
        Returns
        ----------
        ndarray
            Returns 1D array that describes the center of mass for the points in the euclidean object.
        """
        if self.size > 0:
            return np.sum(self.points, axis=0) / self.size
        else:
            return None

    def get_distance(self, metric="euclidean", metric_parameter=None):
        """
        Converts an EucObject into a Distance object by using the specified metric.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". It is a string specifying which metric is used to calculate distances between
            the points of the EucObject. Among possible choices are "euclidean", "cityblock", "cosine", and
            "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if
            "minkowski" metric is chosen, a number p can be specified resulting in the $l_p$ metric.

        Returns
        ----------
        Distance object.
        """
        if self.size == 0:
            return Distance("empty")
        elif self.size == 1:
            return Distance([])
        else:
            return Distance(spatial.distance.pdist(self.points, metric, metric_parameter))

    def filter_distance_to_center(self, center, metric: str="euclidean", metric_parameter=None):
        r"""
        It produces a filter on the self euclidean object $X$ which is a function $f\colon X\to \mathbb{R}$.
        This filter assigns to an element its distance to the center.

        Parameters
        ----------
        center: int, list[Real], ndarray
            either:
                - integer from the list <code>np.arange(self.size)</code>encoding indicating which point of the self
                EucObject is considered, or
                - 1D array like of real numbers whose size equals to the dimension of the object self.
        metric: str
            optional, default "euclidean". It is a string specifying which metric is used to calculate distances between
            the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if "minkowski"
            metric is chosen, a number p can be specified resulting in the $l_p$ metric.
        Returns
        ----------
        1D ndarray
        """
        if isinstance(center, int):
            c = np.array([self.points[center]])
        else:
            c = np.array([center])
        if metric_parameter is None:
            return spatial.distance.cdist(c, self.points, metric)
        else:
            if metric == 'minkowski':
                return spatial.distance.cdist(c, self.points, "minkowski", p=metric_parameter)
            elif metric == 'seuclidean':
                return spatial.distance.cdist(c, self.points, "seuclidean", V=metric_parameter)
            elif metric == "mahalanobis":
                return spatial.distance.cdist(c, self.points, "mahalanobis", VI=metric_parameter)

    def filter_max_distance(self, metric="euclidean", metric_parameter=None):
        r"""
        It produces a filter on the self euclidean object $X$ which is a function $f\colon X\to \mathbb{R}$.
        This filter assigns to an element the max distance among all the distances from this element to all elements in
        $X$.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". It is a string specifying which metric is used to calculate distances between
            the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if "minkowski"
            metric is chosen, a number p can be specified resulting in the $l_p$ metric.

        Returns
        ----------
        1D ndarray
        """
        d = spatial.distance.squareform(spatial.distance.pdist(self.points, metric, metric_parameter), checks=False)
        return np.amax(d, axis=1)

    def filter_proj(self, center, direction):
        r"""
        It produces a filter on the self euclidean object $X$ which is a function $f\colon X\to \mathbb{R}$.
        This filter assigns to an element $x\in X$ the dot product
        $(x-\text{center})\cdot \text{direction}/|\text{direction}|$.

        Parameters
        ----------

        center: int, list[Real], ndarray
            either:
                - integer from the list <code>np.arange(self.size)</code>encoding indicating which point of the self
                EucObject is considered, or
                - 1D array like of real numbers whose size equals  the dimension of the object self.
        direction: list[Real], ndarray
            non-zero 1D array like whose size equals  the dimension of the object self. It represent the direction of
            line we project the data on.

        Returns
        ----------
        1D ndarray
        """
        return np.dot(self.points - center, direction / np.linalg.norm(direction))

    def get_h0sr(self, metric="euclidean", metric_parameter=None, clustering_method="single",
                 contour=Contour(Density([[0], [1]]), "dist", inf)):
        """
        Homology stable ranks are the key invariants of point clouds (subsets of $R^n$) this software aims at
        calculating. The method get_h0sr retrieves the 0-th homology stable rank with respect to a choice of a metric,
        hierarchical clustering method, and contour. A choice of a hierarchical clustering method is not possible
        for higher homologies.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". It is a string specifying which metric is used to calculate distances
            between the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if "minkowski" metric
            is chosen, a number $p$ can be specified resulting in the $l_p$ metric.
        clustering_method: str
            optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        contour: Contour
            optional, default <code>(contour(Density([[0], [1]]), "dist", inf)</code>.

        Returns
        ----------
        Pcnif object.
        """
        return _a_to_h0sr(self.points, metric, metric_parameter, clustering_method, contour)

    def get_bc(self, metric="euclidean", metric_parameter=None, maxdim=1, thresh=inf, coeff=2):
        """
        Homology stable ranks are the key invariants of point clouds (subsets of $R^n$) this software aims at
        calculating. The method get_h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical
        clustering method and a Contour. A choice of a hierarchical clustering method is not possible for higher
        homologies. For higher homologies the Vietoris Rips construction is used instead. The ripser software is the
        computational heart behind extracting bar codes of the homology of the Vietoris Rips construction which is
        the outcome of the get_bc method. The method get_bc retrieves the barcodes of the homologies in degrees not
        exceeding the parameter maxdim.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". It specifies which metric is used to calculate distances between the points.
            Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if
            "minkowski" metric is chosen, a number $p$ can be specified resulting in the $l_p$ metric.
        maxdim: int
            optional, default 1. It is the maximum degree of the calculated homology.
        thresh: extended real number
            optional, default inf. Distances above this threshold are not considered.
        coeff: integer
            optional, default 2. It specifies the characteristic of the field of coefficients with respect to
            which the homology is calculated.

        Returns
        ----------
        Dictionary
            whose keys are strings "Hd" for $0 \\leq d \\leq \\text{maxdim}$ and whose values as BC objects.
        """
        return _a_to_bc(self.points, metric, metric_parameter, maxdim, thresh, coeff)


class SEucObject(object):
    """
    Objects in this class represent pairs consisting of an EucObject and its sampling.

    Attributes
    ----------
    points: ndarray
        2D ndarray of real numbers whose rows represent points of the EucObject.
    size: int
        describing number of points in the EucObject determined by the attribute points. It is the number of rows in
        the attribute points
    dim: int
        integer describing the dimension of the space in which the EucObject is embedded. It is the number of columns
        in the attribute points.
    sample: str, 2D ndarray
        can be either:
            - string "all", or
            - string "empty", or
            - 2D ndarray of integers.
    sample_size: int
        integer which is:
            - equal to the attribute size in case the sample attribute is the string "all".
            - 0 if the sample attribute is the string "empty".
            - the number of columns of the attribute sample in the case this attribute is a 2D ndarray.
    number_instances: int
        integer which is:
            - 0 if the sample attribute is the string "empty",
            - 1 if the sample attribute is the string "all",
            - the number of rows of the attribute sample in the case this attribute is a 2D ndarray.
    """
    def __init__(self, points, sample=None):
        """
        Parameters
        ----------
        points: EucObject, list[list[Real]], ndarray
            either:
                -   EucObject, or
                -   2D array like of real numbers.
        sample: None, str, Sample, list[list[int]], list[list[Real]], ndarray
            optional either:
                - None, default, in which case the sample attribute is set to "all", or
                - string "empty", in which case the sample attribute is set to "empty", or
                - Sample object, in which case the 2D ndarray of integers  <code> s.Sample</code> becomes the attribute
                sample.
                - 2D array like of integers/reals, in that case it is converted to 2D ndarray of integers which becomes
                the attribute sample.
        """
        if isinstance(points, EucObject):
            self.points = points.points
            self.size = points.size
            self.dim = points.dim
        else:
            self.points = np.asarray(points, dtype=float)
            self.size = len(self.points)
            self.dim = self.points.shape[1]
        if sample is None or (isinstance(sample, str) and sample == "all"):
            self.sample = "all"
            self.sample_size = self.size
            self.number_instances = 1
        elif isinstance(sample, str) and sample == "empty":
            self.sample = "empty"
            self.sample_size = 0
            self.number_instances = 0
        elif isinstance(sample, Sample):
            self.sample = sample.sample
            self.sample_size = sample.sample_size
            self.number_instances = sample.number_instances
        else:
            ss = np.asarray(sample, dtype=int)
            if np.shape(ss)[0] == 0 or np.shape(ss)[1] == 0:
                self.sample = "empty"
                self.number_instances = 0
                self.sample_size = 0
            else:
                self.sample = ss
                self.sample_size = ss.shape[1]
                self.number_instances = ss.shape[0]

    def union(self, other):
        """
        This method concatenates the SEuc-objects: by taking the disjoint union of the Euc_objects (it is assumed they
        have the same dimension) and the product of the samplings:
        if (O1,S1) and (O2,S2) are pairs represented by two SEuc_objects, then their union is
        the SEuc_object given by the concatenation of O1 and O2 with the sampling consisting of all the pairs
        (s1,s2) where s1 is a sampling in S1 and s2 is a sampling in S2.

        Parameters
        ----------
        other: SEucObject
            it is the object that is concatenated to the self SEuc_object.

        Returns
        -------
        SEuc_object,
            which is the concatenation of the self and other SEuc_object.
        """
        p1 = self.points
        s1 = self.sample
        p2 = other.points
        s2 = other.sample
        points = np.concatenate([p1, p2])
        if isinstance(s1, str) and s1 == "empty":
            if isinstance(s2, str) and s2 == "empty":
                return SEucObject(points, "empty")
            elif isinstance(s2, str) and s2 == "all":
                s_out = [np.arange(other.size) + self.size]
                return SEucObject(points, s_out)
            else:
                s_out = s2 + self.size
                #i = 0
                #while i < other.number_instances:
                #    s_out[i] = s2.at[i] + self.size
                #    i += 1
                return SEucObject(points, s_out)
        elif isinstance(s1, str) and s1 == "all":
            if isinstance(s2, str) and s2 == "empty":
                s_out = np.arange(self.size)
                return SEucObject(points, s_out)
            elif isinstance(s2, str) and s2 == "all":
                s_out = np.arange(self.size + other.size)
                return SEucObject(points, s_out)
            else:
                #s_out = {}
                w1 = np.array([np.arange(self.size), ]*other.number_instances)
                s_out = np.concatenate([w1, s2+self.size], axis=1)

                #i = 0
                #while i < other.number_instances:
                #    w2 = s2.at[i] + self.size
                #    s_out[i] = np.concatenate([w1, w2])
                 #   i += 1
                return SEucObject(points, s_out)
        else:
            if isinstance(s2, str) and s2 == "empty":
                return SEucObject(points, s1)
            elif isinstance(s2, str) and s2 == "all":
                #s_out = {}
                w2 = np.array([np.arange(other.size) + self.size, ] * self.number_instances)
                s_out = np.concatenate([s1, w2], axis=1)
                #i = 0
                #while i < self.number_instances:
                #    s_out[i] = np.concatenate([s1.at[i], w2])
                #    i += 1
                return SEucObject(points, s_out)
            else:
                new_number_instances = self.number_instances * other.number_instances
                new_sample_size = self.sample_size + other.sample_size
                s_out = np.zeros([new_number_instances, new_sample_size])
                _i = 0
                _k = 0
                while _i < self.number_instances:
                    _j = 0
                    while _j < other.number_instances:
                        s_out[_k] = np.concatenate([s1[_i], s2[_j] + self.size])
                        _j += 1
                        _k += 1
                    _i += 1
                return SEucObject(points, s_out)

    def get_h0sr(self, metric="euclidean", metric_parameter=None, clustering_method="single",
                 contour=Contour(Density([[0], [1]]), "dist", inf)):
        """
        Homology stable ranks are the key invariants of point clouds this software aims at calculating.
        The method get_h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical clustering
        method and a contour. A choice of a hierarchical clustering method is not possible for higher homologies.
        Depending on self.Sample object, a global or averaged stable rank is returned:
        -   If self.Sample is "all": a Pcnif object is returned representing the stable rank of the EucObject
            represented by self.points. The stable rank is calculated with respect to the specified metric,
            clustering_method, and contour.
        -   If self.Sample is "empty": the 0 Pcnif object <code>Pcnif([[0],[0]])</code> is returned.
        -   If  self.Sample is a 2D ndarray: a Pcnif object is returned representing the average
            stable rank constructed as follows:
            -   for each row of self.Sample take the subspace of the self.points determined by the coordinates
                in the row,
            -   then calculate the stable rank of this subspace with respect to the chosen clustering method
                and contour,
            -   finally return the average stable rank across all rows of self.Sample.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". It specifies which metric is used to calculate distances between the points.
            Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameter for the metric. For example if
            "minkowski" metric is chosen, a number $p$ can be specified resulting in the $l_p$ metric.
        clustering_method: str
            optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        contour: Contour
            optional, default <code>(contour(Density([[0], [1]]), "dist", inf)</code>.

        Returns
        -------
        Pcnif object.
        """
        if isinstance(self.sample, str) and self.sample == "all":
            return _a_to_h0sr(self.points, metric, metric_parameter, clustering_method, contour)
        elif isinstance(self.sample, str) and self.sample == "empty":
            return Pcnif([[0], [0]])
        else:
            f = Pcnif([[0], [0]])
            inst = 0
            while inst < self.number_instances:
                ind = self.sample[inst]
                g = _a_to_h0sr(self.points[ind], metric, metric_parameter, clustering_method, contour)
                f = f + g
                inst += 1
            return f * (1/self.number_instances)

    def get_bc(self, metric="euclidean", metric_parameter=None, maxdim=1, thresh=inf, coeff=2):
        """
        Homology stable ranks are the key invariants of point clouds this software aims at calculating.
        The method get_h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical clustering
        method and a Contour. A choice of a hierarchical clustering method is not possible for higher homologies.
        For higher homologies the Vietoris Rips construction is used instead. The ripser software is the computational
        heart behind extracting bar codes of the homology of the Vietoris Rips construction which is the outcome of
        the get_bc method. Depending on self.Sample the following is returned:
        -   If self.Sample is "empty": a dictionary is returned whose keys are strings "Hd"
            for $0 \\leq d \\leq \\text{maxdim}$.  Its values are the empty bar codes <code>BC([])</code>.
        -   If self.Sample is "all": a dictionary is returned whose keys are strings "Hd"
            for $0 \\leq d \\leq \\text{maxdim}$. Its values are the bar-codes of the EucObject
            described by self.points.
        -   If self.Sample is 2D ndarray of integers: a dictionary oif dictionaries is returned. Its keys are strings
            "Hd" for $0 \\leq d \\leq \\text{maxdim}$. For each such key "Hd", the value is a dictionary
            whose keys are integers in <code>range(self.number_instances)</code>. For each such key i the value is
            the "Hd" barcode of  the subspace of the self.points determined by the coordinates of i-th raw of
            the i-th row of self.Sample.

        Parameters
        ----------
        metric: str
            optional, default "euclidean". A string specifying which metric is used to calculate distances between
            the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
        metric_parameter: None, Real
            optional, default None. It specifies possible parameters for the metric. For example if
            "minkowski" metric is chosen, a number $p$ can be specified resulting in the $l_p$ metric.
        maxdim: int
            optional, default 1. It is the maximum degree of the calculated homology.
        thresh: extended real number
            optional, default inf. Distances above this threshold are not considered.
        coeff: integer
            optional, default 2. It specifies the characteristic of the field of coefficients with respect to
            which the homology is calculated.

        Returns
        ----------
        Dictionary:
            in case self.Sample is either "empty" or "all". Its  keys are strings "Hd"
            for $0 \\leq d \\leq \\text{maxdim}$.
        Dictionary of dictionaries
            in case self.Sample is a Sample object. The keys are strings "Hd"
            for $0 \\leq d \\leq \\text{maxdim}$. For each such key "Hd", the value is a dictionary
            whose keys are integers in <code>range(self.number_instances)</code>. For each such key i the value is
            the "Hd" barcode of of the subspace of the self.points determined by the coordinates of i-th raw of
            the
            the value self.Sample at the considered index.

            Its index, called homology, is given by the list
            <code>["H"+str(d) for d in np.arange(maxdim+1)]</code>. Its Columns, called
            instances, are given the index of self.Sample.
        """

        if isinstance(self.sample, str) and self.sample == "all":
            return _a_to_bc(self.points, metric, metric_parameter, maxdim, thresh, coeff)
        elif isinstance(self.sample, str) and self.sample == "empty":
            return _a_to_bc([], metric, metric_parameter, maxdim, thresh, coeff)
        else:
            out = {}
            inst = 0
            _b = {}
            while inst < self.number_instances:
                ind = self.sample[inst]
                _b[inst] = _a_to_bc(self.points[ind], metric, metric_parameter, maxdim, thresh, coeff)
                inst += 1
            _d = 0
            while _d <= maxdim:
                _h = "H"+str(_d)
                out[_h] = {}
                inst = 0
                while inst < self.number_instances:
                    out[_h][inst] = _b[inst][_h]
                    inst += 1
                _d += 1
            return out


def _a_to_bc(a, metric, metric_parameter, maxdim, thresh, coeff):
    """a is assumed to be 2D array like"""
    if len(a) == 0:
        return empty_bc(maxdim)
    elif len(a) == 1:
        return one_point_bc(maxdim)
    else:
        d = spatial.distance.squareform(_pdist(np.asarray(a), metric, metric_parameter), checks=False)
        dgms = ripser(d, maxdim=maxdim, thresh=thresh, coeff=coeff, distance_matrix=True, do_cocycles=False)["dgms"]
        out = {"H" + str(h): BC(dgms[h]) for h in range(maxdim + 1)}
        return out


def _linkage_to_stable_rank(link, contour):
    """contour is a Contour object"""
    d = np.array([0])
    val = np.array([len(link) + 1])
    ind = 0
    j = 0
    s = len(link) + 1
    while ind < len(link[:, 2]):
        th = link[ind, 2]
        bar = contour.density.bar_length([0, th], contour.kind, contour.truncation)
        s = s - 1
        if th > d[j]:
            d = np.append(d, bar)
            val = np.append(val, s)
            j += 1
        else:
            val[j] = val[j] - 1
        ind += 1
    return Pcnif(np.vstack((d, val)))


def _a_to_h0sr(a, metric, metric_parameter, clustering_method, contour):
    """a is assumed to be 2D array like"""
    if len(a) == 0:
        return Pcnif([[0], [0]])
    elif len(a) == 1:
        return Pcnif([[0], [1]])
    else:
        d = _pdist(a, metric, metric_parameter)
        link = linkage(d, clustering_method)
        f = _linkage_to_stable_rank(link, contour)
        return f


def _pdist(a, metric, metric_parameter=None):
    if metric is None:
        return spatial.distance.pdist(a, "euclidean")
    else:
        if metric_parameter is None:
            return spatial.distance.pdist(a, metric)
        if metric == 'minkowski':
            return spatial.distance.pdist(a, "minkowski", p=metric_parameter)
        if metric == 'seuclidean':
            return spatial.distance.pdist(a, "seuclidean", V=metric_parameter)
        if metric == "mahalanobis":
            return spatial.distance.pdist(a, "mahalanobis", VI=metric_parameter)
        ValueError("no such metric")
