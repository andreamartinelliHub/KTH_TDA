#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:47:52 2021

@author: wojtek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial as spatial

from ripser import ripser

from stablerank.rtorf import Pcf
from stablerank.rtorf import Density
from stablerank.rtorf import Pcnif
from stablerank.rtorf import Distribution
from stablerank.rtorf import Contour
from stablerank.rtorf import moving_distribution
from stablerank.rtorf import increasing_distribution
from stablerank.rtorf import decreasing_distribution

from stablerank.functions import build_pd_Series
from stablerank.functions import randomchoice

from stablerank.covers import Cover
from stablerank.covers import cover_product

from stablerank.distances import Distance
from stablerank.distances import SDistance


from stablerank.eucobjects import EucObject
from stablerank.eucobjects import SEucObject

from stablerank.barcodes import BC

from stablerank.sample import Sample
from stablerank.sample import get_sample

inf = float("inf")


def get_filter(point, ref_object, filter_type: str = "distance", metric=None,
               metric_parameter=None, center=None, vector_field=None):
    """
    Parameters
    ----------
    point: list[Real], ndarray
        1D array like of real numbers.
    ref_object: list[list[Real]], ndarray, EucObject
        Could be either:
        - 2D array like of reals, or
        - EucObject
    filter_type: str
        could be either "distance" or "proj_center" or "proj_dir".
    metric: str
        string, optional, default "euclidean". This parameter is relevant only in the case
        the parameter filter_type is "distance". It specifies which metric is used to
        calculate distances between the point and rows of ref_objects. Among possible
        choices are "euclidean", "cityblock", "cosine", and "minkowski".
    metric_parameter: None, additional parameter needed to specify the metric.
        optional, default None. It is relevant only if the parameter metric  is 'minkowski',
        or 'seuclidean', or "mahalanobis".
    center: None, list[Real], ndarray
        optional, default None. It is relevant only in the case the parameter filter_type
        is "proj_center". If None, then the center of mass of ref_object is taken as
        the center. Otherwise it is 1D array like of real numbers of the same length
        as the parameter point.
    vector_field: list[Real], list[list[Real]], ndarray.
        It is either 1D or 2D array like of real numbers. In the 1D case it has the same
        length as the parameter point. In the 2D case, the number of rows has to be the
        same as  number of rows of the parameter ref_object, and number of columns is
        the same as the length as the parameter point. It is relevant only in the case
        the parameter filter_type is "proj_dir".

    Returns
    -------
    1D ndarray of length number of rows of the parameter ref_object.
    """
    if isinstance(ref_object, EucObject):
        ref_points = ref_object.points
    else:
        ref_points = ref_object
    if filter_type == "distance":
        return _cdist(point, ref_points, metric, metric_parameter)
    number_points = len(ref_points)
    outcome = np.empty(number_points)
    if filter_type == "proj_center":
        if center is None:
            center = np.sum(ref_points, axis=0) / number_points
        i = 0
        while i < number_points:
            direction = center - ref_points[i]
            outcome[i] = np.dot(point - ref_points[i], direction / np.linalg.norm(direction))
            i += 1
        return outcome
    if filter_type == "proj_dir":
        if np.ndim(vector_field) == 1:
            i = 0
            while i < number_points:
                outcome[i] = np.dot(point - ref_points[i],
                                    vector_field / np.linalg.norm(vector_field))
                i += 1
            return outcome
        else:
            i = 0
            while i < number_points:
                outcome[i] = np.dot(point - ref_points[i],
                                    vector_field[i] / np.linalg.norm(vector_field[i]))
                i += 1
            return outcome
    raise ValueError("Wrong parameters")


def get_h0sr(in_object, metric="euclidean", metric_parameter=None, clustering_method="single",
             contour=Contour(Density([[0], [1]]), "dist", inf)):
    """
    Homology stable ranks are the key invariants of point clouds this software aims at calculating. The method h0sr
    retrieves the 0-th homology stable rank with respect to a chosen metric, hierarchical clustering method, and
    a contour.

    Parameters
    ----------
    in_object: Distance, SDistance, EucObject, SEucObject, list[list[Real]], ndarray, dict
        Could be either:
        -   Distance object, or
        -   SDistance object, or
        -   EucObject, or
        -   SEucObject, or
        -   2D array like of real numbers.
    metric: str
        string, optional, default "euclidean". This parameter is only relevant in the case the parameter in_object is
        either EucObject, or SEucObject, or 2D array like. It specifies which metric is used to calculate distances
        between the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
    metric_parameter: None, Real
        optional, default None. It specifies possible parameters for the metric. For example if
        "minkowski" metric is chosen, a number $p$ can be specified resulting in the $l_p$ metric.
    clustering_method: str
        optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
        Among possible choices are "single", "complete", "average", and "ward".
    contour: Contour
        optional, default <code>(contour(Density([[0], [1]]), "dist", inf)</code>.

    Returns
    ----------
    Pcnif object.
    """
    if isinstance(in_object, (Distance, SDistance)):
        return in_object.get_h0sr(clustering_method, contour)
    elif isinstance(in_object, (EucObject, SEucObject)):
        return in_object.get_h0sr(metric, metric_parameter, clustering_method, contour)
    else:  # in_object is assumed to be 2D array like
        return _a_to_h0sr(in_object, metric, metric_parameter, clustering_method, contour)


def get_bc(in_object, metric="euclidean", metric_parameter=None, maxdim=1, thresh=inf, coeff=2):
    """
    Homology stable ranks are the key invariants of point clouds this software aims at calculating.
    The method h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical clustering
    method and a contour. A choice of a hierarchical clustering method is not possible for higher homologies.
    For higher homologies the Vietoris Rips construction is used instead. The ripser software is the computational
    heart behind extracting bar codes of the homology of the Vietoris Rips construction which is the outcome of
    the bc method. The output is a dictionary whose keys are strings "Hd" for $0 \\leq d \\leq \\text{maxdim}$.
    Depending on in_object its values are either BC objects, in the case the in_object is of Distantce, EucObject or
    array like type, or a dictionary of such objects in the case the in_object is of SDistance or SEucObject type.
    Its keys are called instances.

    Parameters
    ----------
    in_object: Distance, SDistance, EucObject, SEucObject, list[list[Real]], ndarray, dict
        Could be either:
        -   Distance object, or
        -   SDistance object, or
        -   EucObject, or
        -   SEucObject, or
        -   2D array like of real numbers.
    metric: str
        string, optional, default "euclidean". This parameter is only relevant in the case the parameter in_object is
        either EucObject or 2D array like or a dictionary. It specifies which metric is used to calculate distances
        between the points. Among possible choices are "euclidean", "cityblock", "cosine", and "minkowski".
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
    -------
    dictionary
        whose values are BC objects and whose keys are strings "Hd" for $0 \\leq d \\leq \\text{maxdim}$.
        This is in the case in_object is a Distance object, or an EucObject or 2D array like.
    dictionary
        whose values are dictionaries of BC objects and whose keys are strings "Hd" for
        $0 \\leq d \\leq \\text{maxdim}$. This is in the case in_object is a SDistance object,
        or an SEucObject. For each such key, the values is a dictionary whose keys are called instances.
    """
    if isinstance(in_object, (Distance, SDistance)):
        return in_object.get_bc(maxdim, thresh, coeff)
    elif isinstance(in_object, (EucObject, SEucObject)):
        return in_object.get_bc(metric, metric_parameter, maxdim, thresh, coeff)
    else:  # in_object is assumed to be 2D array like
        return _a_to_bc(in_object, metric, metric_parameter, maxdim, thresh, coeff)


def bc_to_sr(bar_code, degree="H1", contour=Contour(Density([[0], [1]]), "dist", inf)):
    """
    This method converts bar_codes into stable ranks with respect to the specified contour.
    Parameters
    ----------
    bar_code: Series, DataFrame
        assumed to be the outcome of methods get_bc for Distance, SDistance,  EucObject, or SEucObject
        or the outcome of the global bc method.
    degree: str
        optional, default "H1", specifies which homology we concentrate on. It is assumed that this string is key
        the <code>bar_code</code>.
    contour: Contour
        optional, default <code>(contour(Density([[0], [1]]), "dist", inf)</code>.
    """
    b = bar_code[degree]
    if isinstance(b, BC):
        return b.stable_rank(c=contour)
    else:
        f = Pcnif([[0], [0]])
        for inst in b.keys():
            g = b[inst].stable_rank(c=contour)
            f += g
        return f * (1 / len(b.keys()))


#################################################################
#################################################################


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


def _a_to_bc(a, metric, metric_parameter, maxdim, thresh, coeff):
    """a is assumed to be 2D array like"""
    if len(a) == 0:
        return empty_bc(maxdim)
    elif len(a) == 1:
        return one_point_bc(maxdim)
    else:
        d = spatial.distance.squareform(spatial.distance.pdist(np.asarray(a), metric, metric_parameter), checks=False)
        dgms = ripser(d, maxdim=maxdim, thresh=thresh, coeff=coeff, distance_matrix=True, do_cocycles=False)["dgms"]
        out = {"H" + str(h): BC(dgms[h]) for h in range(maxdim + 1)}
        return out


def _cdist(p, a, metric=None, metric_parameter=None):
    if metric is None:
        return spatial.distance.cdist(np.array([p]), a, "euclidean")[0]
    else:
        if metric_parameter is None:
            return spatial.distance.cdist(np.array([p]), a, metric)[0]
        if metric == 'minkowski':
            return spatial.distance.cdist(np.array([p]), a, "minkowski", p=metric_parameter)[0]
        if metric == 'seuclidean':
            return spatial.distance.cdist(np.array([p]), a, "seuclidean", V=metric_parameter)[0]
        if metric == "mahalanobis":
            return spatial.distance.cdist(np.array([p]), a, "mahalanobis", VI=metric_parameter)[0]
        ValueError("no such metric")


def _pdist(a, metric=None, metric_parameter=None):
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





# def _part_dict(in_object):
#     dim = list(in_object.values())[0].dim
#     k_all = [k for k in in_object.keys() if isinstance(in_object[k].sample, str) and in_object[k].sample == "all"]
#     k_1 = [k for k in in_object.keys() if not isinstance(in_object[k].sample, str) and
#            in_object[k].number_instances == 1]
#     k_2 = [k for k in in_object.keys() if in_object[k].number_instances > 1]
#     a = np.empty([0, dim])
#     if len(k_all) > 0:
#         a = np.concatenate([in_object[k].points for k in k_all])
#     if len(k_1) > 0:
#         b = np.concatenate([in_object[k].points[in_object[k].sample[0]] for k in k_1])
#         a = np.concatenate([a,b])
#     return {"k_2": k_2, "a": a}
#
#
#
#
#
# def _mseuc_obj_get_bc(in_object, metric, metric_parameter, maxdim, thresh, coeff):
#     """in_object is a dictionary of SEucObject's"""
#     part = _part_dict(in_object)
#     k_2 = part["k_2"]
#     a = part["a"]
#     if len(k_2) == 0:
#         return _a_to_bc(a, metric, metric_parameter, maxdim, thresh, coeff)
#     else:
#         j = [np.arange(in_object[k].number_instances) for k in k_2]
#         columns = pd.MultiIndex.from_product(j)
#         out = {}
#         for col in columns:
#             w = [in_object[k_2[p]].points[in_object[k_2[p]].sample[col[p]]] for p in range(len(k_2))]
#             c = np.concatenate(w)
#             out[col] = _a_to_bc(np.concatenate([a, c]), metric, metric_parameter, maxdim, thresh, coeff)
#         out = pd.DataFrame(out, dtype=object)
#         out.columns.name = "instances"
#         return out


# def _series_cover_to_cover(cover):
#     """cover is assumed to be a panda Series"""
#     cov = {}
#     ob = []
#     empty = []
#     for _i in cover.index:
#         _t = tuple(cover.at[_i])
#         cov[_i] = _t
#         ob.extend(list(cover.at[_i]))
#         if len(_t) == 0:
#             empty.append(_i)
#     ob = tuple(set(ob))
#     ind = tuple(set(cover.index))
#     cont = {}
#     for o in ob:
#         c = []
#         for b in ind:
#             if o in cover.at[b]:
#                 c.append(b)
#         cont[o] = tuple(c)
#     cont["empty_blocks"] = tuple(empty)
#     return {"cover": cov, "observations": ob, "index": ind, "containers": cont}
#
#
# def _dict_cover_to_cover(cover):
#     """cover is assumed to be a dictionary"""
#     cov = {}
#     ob = []
#     empty = []
#     for _i in cover.keys():
#         _t = tuple(cover[_i])
#         cov[_i] = _t
#         ob.extend(list(cover[_i]))
#         if len(_t) == 0:
#             empty.append(_i)
#     ob = tuple(set(ob))
#     ind = tuple(set(cover.keys()))
#     cont = {}
#     for o in ob:
#         c = []
#         for b in ind:
#             if o in cover[b]:
#                 c.append(b)
#         cont[o] = tuple(c)
#     cont["empty_blocks"] = tuple(empty)
#     return {"cover": cov, "observations": ob, "index": ind, "containers": cont}
#
# # def _series_cover_to_cover(cover):
# #     """cover is assumed to be a panda Series"""
# #     cov = cover.apply(lambda _x: tuple(_x))
# #     ob = []
# #     for _i in cover.index:
# #         ob.extend(list(cover.at[_i]))
# #     ob = tuple(set(ob))
# #     ind = tuple(set(cover.index))
# #     cont = {}
# #     for o in ob:
# #         c = []
# #         for b in ind:
# #             if o in cover.at[b]:
# #                 c.append(b)
# #         cont[o] = tuple(c)
# #     v = cover.apply(lambda x: len(x))
# #     cont["empty_blocks"] = tuple(v[v == 0].index)
# #     cont = pd.Series(cont, dtype=object)
# #     return {"cover": cov, "observations": ob, "index": ind, "containers": cont}
#
#
# # def _dict_cover_to_cover(cover):
# #     """cover is assumed to be a dictionary"""
# #     return _series_cover_to_cover(pd.Series(cover, dtype=object))
#
#
# def _dict_series_cover(cover):
#     """cover is assumed to be a dictionary of panda Serieses"""
#     cov = {}
#     for _i in cover.keys():
#         for _j in cover[_i].index:
#             cov[(_i, _j)] = cover[_i].at[_j]
#     return Cover(cover=cov)
#
#
# def _series_containers_to_cover(containers):
#     """containers is assumed to be a panda Series"""
#     ob = tuple([x for x in containers.index if x != "empty_blocks" and len(containers.at[x]) > 0])
#     cont = {}
#     cov = {}
#     ind = []
#     for x in ob:
#         ind.extend(list(containers.at[x]))
#         cont[x] = tuple(containers.at[x])
#     if "empty_blocks" in containers.index:
#         ind.extend(list(containers.at["empty_blocks"]))
#         cont["empty_blocks"] = containers.at["empty_blocks"]
#         for _i in containers.at["empty_blocks"]:
#             cov[_i] = tuple([])
#     else:
#         cont["empty_blocks"] = tuple([])
#     ind = tuple(set(ind))
#     for _i in ind:
#         block = []
#         for x in ob:
#             if _i in containers.at[x]:
#                 block.append(x)
#         cov[_i] = tuple(block)
#     return {"cover": cov, "observations": ob, "index": ind, "containers": cont}
#
#
# def _dict_containers_to_cover(containers):
#     """containers is assumed to be a dictionary"""
#     ob = tuple([x for x in containers.keys() if x != "empty_blocks" and len(containers[x]) > 0])
#     cont = {}
#     cov = {}
#     ind = []
#     for x in ob:
#         ind.extend(list(containers[x]))
#         cont[x] = tuple(containers[x])
#     if "empty_blocks" in containers.keys():
#         ind.extend(list(containers["empty_blocks"]))
#         cont["empty_blocks"] = containers["empty_blocks"]
#         for _i in containers["empty_blocks"]:
#             cov[_i] = tuple([])
#     else:
#         cont["empty_blocks"] = tuple([])
#     ind = tuple(set(ind))
#     for _i in ind:
#         block = []
#         for x in ob:
#             if _i in containers[x]:
#                 block.append(x)
#         cov[_i] = tuple(block)
#     return {"cover": cov, "observations": ob, "index": ind, "containers": cont}
#
#
# def _partition_to_cover(partition, labels=None):
#     if labels is None:
#         labels = range(len(partition))
#     cont = {}
#     _i = 0
#     for _l in labels:
#         cont[_l] = tuple([partition[_i]])
#         _i += 1
#     return _dict_containers_to_cover(cont)


# def _interval_subdivide(b, e, n, overlap):
#     a = np.zeros([n, 2])
#     s = (e - b) / n
#     a[0] = [b, b + s + (s * overlap) / 100]
#     _i = 1
#     while _i < n - 1:
#         a[_i] = [b + s*_i - (s*overlap)/100, b + s*(_i + 1) + (s*overlap)/100]
#         _i += 1
#     a[n - 1] = [b + (n - 1)*s - (s*overlap)/100, e]
#     return a


# def _filter_subdivide(f, number_intervals, overlap):
#     b = np.amin(f)
#     e = np.amax(f)
#     intervals = _interval_subdivide(b, e, number_intervals, overlap)
#     cov = {}
#     _i = 0
#     while _i < len(intervals):
#         start = intervals[_i][0]
#         end = intervals[_i][1]
#         cov[_i] = tuple(np.where(np.logical_and(f >= start, f <= end))[0])
#         _i += 1
#     return Cover(cover=cov)

####################################################################################
####################################################################################
####################################################################################






