#!/usr/bin/env python3
"""
Created on Sun June 13 18:27 2021

@author: wojtek
"""
import numpy as np
import pandas as pd
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

from stablerank.sample import Sample
from stablerank.sample import get_sample

from stablerank.covers import Cover


inf = float("inf")


class Distance(object):
    """
    Objects in this class represent finite distance spaces. To define a Distance object, we need to specify either 1D
    array like object (condensed form) or a square symmetric 2D array like object with 0 on the diagonal. In the case
    of the 2D array, only the entries above the diagonal are considered. The entries of this array are assumed to be
    non-negative and represent distances between points in the distance space.

    Attributes
    ----------
    size: int
        integer which is the size of the distance space.
    content: string, ndarray
        either string "empty" or 1D ndarray of non-negative extended real numbers describing the condense form of
        the distance space. Content is "empty" if and only if the attribute size is 0.
    limit: Real
        extended real number used as follows. Every inf entry in content is converted to the biggest finite entry times
        the limit. This is relevant only if content is ndarray.
    """
    def __init__(self, d, limit=2):
        """
        Parameters
        ----------
        d: list[Real], list[list[Real]], ndarray, str
            either:
            -   1D array like of non-negative extended real numbers, or
            -   a square array like of non-negative extended real numbers with 0 on the diagonal, or
            -   string "empty".
        limit: Real
            extended real number optional, default 2. Any inf entry in d is converted to the biggest finite entry
            times the limit parameter.
        """
        if isinstance(d, str):
            self.size = 0
            self.content = "empty"
        else:
            den = np.asarray(d, dtype=float)
            if np.ndim(den) == 1 and len(den) == 0:
                self.size = 1
                self.content = np.array([])
            elif np.ndim(den) == 2 and den.size == 0:
                self.size = 0
                self.content = "empty"
            else:
                m = np.amax(den[den != inf])
                den[den == inf] = limit * m
                if np.ndim(den) == 1:
                    self.size = int((1 + np.sqrt(1 + 8 * len(den))) / 2)
                    self.content = den[0:int(self.size * (self.size - 1) / 2)]
                else:
                    self.size = int(den.shape[0])
                    self.content = spatial.distance.squareform(den, checks=False)
        self.limit = limit

    def square_form(self):
        r"""
        Produces a square ndarray whose entries encode distances.

        Example
        -------
        $\begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6  \end{bmatrix} \mapsto \begin{bmatrix}0 & 1 & 2 & 3\\\\ 1 & 0 & 4 & 5\\\\
        2 & 4 &0 & 6 \\\\ 3 & 5 & 6 & 0 \end{bmatrix}$

        Returns
        ------
        square 2D ndarray
            whose entries  are real numbers which are non-negative with 0 on the diagonal.
            These entries are the distances between the points in the represented distance space.
        """
        if isinstance(self.content, str):
            return np.empty([0,0])
        else:
            return spatial.distance.squareform(self.content, checks=False)

    def diameter(self, proportion=0.9, number_instances=200):
        """
        Produces a real number constructed as follows: first choose randomly proportion times self.size number of
        elements in the distance space. Then take the maximum distance among the chosen points which is the diameter of
        this subspace. Repeat this procedure number_instances times and take the minimum among the obtained
        diameters.

        Parameters
        ----------
        proportion: Real
            real number between 0 and 1.
        number_instances: int
            integer describing how many times the distance space is sampled.

        Returns
        -------
        Real.
        """
        if self.size <= 1:
            return 0.
        else:
            s_s = int(proportion * self.size)
            square_form = self.square_form()
            out = np.zeros(number_instances)
            i = 0
            while i < number_instances:
                si = np.sort(np.random.choice(self.size, s_s, replace=False))
                out[i] = np.amax(square_form[np.ix_(si, si)])
                i += 1
            return np.amin(out)

    def filter_distance_to_center(self, center):
        r"""
        It produces a filter on the self distance object $X$ which is a function $f\colon X\to \mathbb{R}$.
        This filter assigns to an element its distance to the center.

        Parameters
        ----------
        center: int
            integer from the list <code>np.arange(self.size)</code>encoding indicating which point of the self
        Returns
        ----------
        1D ndarray
        """
        return self.square_form()[center]

    def filter_max_distance(self):
        r"""
        It produces a filter on the self distance object $X$ which is a function $f\colon X\to \mathbb{R}$.
        This filter assigns to an element the max distance among all the distances from this element to all elements in
        $X$.

        Returns
        ----------
        1D ndarray
        """
        return np.amax(self.square_form(), axis=1)

    def get_h0sr(self, clustering_method="single", contour=Contour(Density([[0], [1]]), "dist", inf)):
        """
        Homology stable ranks are the key invariants of point clouds this software aims at calculating.
        The method get_h0sr returns the 0-th homology stable rank with respect to a chosen hierarchical
        clustering method and a contour. A choice of hierarchical clustering method is not possible
        for higher homologies.

        Parameters
        ----------
        clustering_method: str
            string, optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        contour: Contour
            contour object optional, default <code>(contour(Density([[0], [1]]), "dist", inf)</code> ,which is an
            essential ingredient in calculating the length of bars.

        Returns
        ----------
        Pcnif object.
        """
        if self.size == 0:
            return Pcnif([[0], [0]])
        else:
            return _d_to_h0sr(self.content, clustering_method, contour)

    def get_bc(self, maxdim=1, thresh=inf, coeff=2):
        r"""
        Homology stable ranks are the key invariants of distance spaces this software aims at calculating.
        The method get_h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical clustering
        method and a Contour. A choice of a hierarchical clustering method is not possible for higher homologies.
        For higher homologies the Vietoris Rips construction is used instead. The ripser software is the computational
        heart behind extracting bar codes of the homology of the Vietoris Rips construction which is the outcome of
        the get_bc method. The method get_bc retrieves the barcodes of the homologies in degrees not exceeding the
        parameter maxdim.

        Parameters
        ----------
        maxdim: int
            integer, optional, default 1, specifying the maximum degree of the calculated homology.
        thresh: Real
            extended real number optional, default inf. Distances above this threshold are not considered.
        coeff: int
            prime integer optional, default 2, specifying the characteristic of the field of coefficients with respect
            to which the homology is calculated.

        Returns
        ----------
        dictionary
            whose keys are strings "Hd" for $0 \leq d \leq \text{maxdim}$.
        """
        if self.size == 0:
            return empty_bc(maxdim)
        else:
            return _d_to_bc(self.square_form(), maxdim, thresh, coeff)

    def partition(self, clustering_method="complete", cutoff=("single", 200, 10), criterion="distance",
                  link_record=None):
        """
        This method partitions the set <code>set(range(self.size))</code> as follows:
        -   if self is the empty Distance object, then the empty array is returned,
        -   if self is of size 1, then <code>np.array([1])</code> is returned,
        -   if self is of size bigger than 1, then the partition is obtained through the method
        <code>fcluster(link, cutoff, criterion)</code> where:
            - link is the linkage constructed with respect to the specified clustering_method via <code>
            linkage(self.content, clustering_method)</code>.
            - if the cutoff parameter is a real number or integer, then it is used as
            the cutoff parameter in the fcluster method.
            - if the cutoff parameter is the string "diff", then h0sr is constructed from the link and
            the beginning of the longest interval over which this h0sr has a constant value is chosen to be
            the cutoff parameter in the fcluster method. In  this case criterion parameter of the fcluster method
            is chosen to be "distance"
            - if the cutoff parameter is a tuple, then its first component, called cutoff_method, is a
            string "single", "complete", "average", or "ward"; its second component, called number_instances, is an
            integer describing how mny samplings are going to be performed; and its third component, called sample size,
            describes the sample size.
            In this case the criterion parameter of the fcluster method is chosen to be "distance" and its cutoff
            parameter is chosen as follows. The average h0sr of the self objects is considered with respect to the
            uniform sampling of sample_size and number_instances times. In this case the cutoff is given by the
            flatness of this average h0sr.

        Parameters
        ----------
        clustering_method: str
            optional, default "complete". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        cutoff: real, int, string, tuple
            optional, default ("single", 200, 10). If it is a real number or int, it plays the role of the cutoff
            parameter in the method <code>fcluster(link, cutoff, criterion)</code>. If it s a tuple, then its first
            component, called cutoff_method, can be either the string "single", or "complete", "average", or "ward";
            its second component, called number_instances, is an integer describing how mny samplings are going to
            be performed; and its third component, called sample size, describes the sample size.
        criterion: str
            optional, default "distance". It is the criterion to use in forming flat clusters using
            <code>fcluster()</code> method. Possible choices are "distance", "maxclust", or "inconsistent".
        link_record: None, str
            optional, default None. If not None, then the link is returned also.

        Returns
        ----------
        dict
            which assigns to the key "partition" a 1D ndarray of integers describing the partition.
            If link is not None, then the link <code> linkage(self.content, clustering_method)</code> is
            assigned to the key "link".
        """
        out = {}
        if isinstance(self.content, str) and self.content == "empty":
            out["partition"] = np.array([])
        elif self.size == 1:
            out["partition"] = np.array([1])
        else:
            link = linkage(self.content, clustering_method)
            if link_record is not None:
                out["link"] = link
            if isinstance(cutoff, (tuple, list)):
                s = get_sample(cutoff[1], cutoff[2], ["uniform", self.size])
                f = SDistance(self, s).get_h0sr(clustering_method=cutoff[0])
                c = f.flatness()[0]
                out["partition"] = fcluster(link, c, criterion='distance')
            elif isinstance(cutoff, str) and cutoff == "diff":
                f = _linkage_to_stable_rank(link, contour=Contour(Density([[0], [1]]), "dist", inf))
                c = f.flatness()[0]
                out["partition"] = fcluster(link, c, criterion='distance')
            else:
                out["partition"] = fcluster(link, cutoff, criterion=criterion)
        return out

    def graph(self, f, number_intervals, overlap, b=None, e=None, clustering_method="complete",
              cutoff=("single", 200, 10), criterion="distance", link_record=None):
        """
        It produces a Cover object representing a Reeb's graph of the filter function represented by the parameter f
        and is constructed as follows:
        - subdivide the interval [<code>amin(f)</code>, <code>amax(f)</code>] into number_intervals of sub-intervals
        that intersect overlap portion.
        - consider the points of the self Distance object for which the value of f belongs to one of the chosen
        sub intervals.
        - partition this subspace of points using the method <code>partition(self, cutoff, criterion)</code>.
        - return the cover given by the union of all these partitions.

        Parameters
        ----------
        f: ndarray, list[real]
            array like of length <code>self.size</code>. It describes a filter function which assigns the i-th point
            in the distance space the i-th value of f.
        number_intervals: int
            specifies into how many intervals the range of the filter function is going to be subdivided.
        overlap: real, int
            specifies the percent of the overlap of the consecutive intervals.
        b: real
            optional, default None, it specifies the beginning of the interval over which the values of the filter
            function are considered. If None, b is chosen to me min(f).
        e: real
            optional, default None, it specifies the end of the interval over which the values of the filter
            function are considered. If None, b is chosen to me max(f).
        clustering_method: str
            optional, default "complete". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        cutoff: real, int, string, tuple
            optional, default ("single", 200, 10). If it is a real number or int, it plays the role of the cutoff
            parameter in the method <code>fcluster(link, cutoff, criterion)</code>. If it s a tuple, then its first
            component, called cutoff_method, can be either the string "single", or "complete", "average", or "ward";
            its second component, called number_instances, is an integer describing how mny samplings are going to
            be performed; and its third component, called sample size, describes the sample size.
        criterion: str
            optional, default "distance". It is the criterion to use in forming flat clusters using
            <code>fcluster()</code> method. Possible choices are "distance", "maxclust", or "inconsistent".
        link_record: None, str
            optional, default None. If not None, then the links of the local clustering are returned.

        Returns
        ----------
        dict
            which assigns to the key "cover" the given by the blocks of partitions of various subspaces determined by
            the filter function. If the parameter link_record is not None, then the a dictionary is assigned to the key
            "links". This dictionary is indexed by subdivisions of the values of the filter function and for each such
            value the corresponding link obtained by <code> linkage(self.content, clustering_method)</code> is assigned.
        """
        if isinstance(self.content, str) and self.content == "empty":
            return {"cover": Cover(cover={})}
        elif self.size == 1:
            return {"cover": Cover(cover={0: [0]})}
        else:
            if b is None:
                b = np.amin(f)
            if e is None:
                e = np.amax(f)
            subdivision = _filter_subdivide(f, number_intervals, overlap, b, e)
            d = self.square_form()
            out = {}
            if link_record is None:
                pass
            else:
                out["links"] = {}
            cov = {}
            for _i in subdivision.index:
                d_i = d[np.ix_(subdivision.cover[_i], subdivision.cover[_i])]
                p = Distance(d_i).partition(clustering_method, cutoff, criterion)
                partition_i = p["partition"]
                cov[_i] = Cover(partition=partition_i, labels=subdivision.cover[_i]).cover
                if link_record is not None:
                    out["links"][_i] = p["link"]
            out["cover"] = _dict_dict_cover(cov)
            return out


class SDistance(object):
    """
    Objects in this class represent pairs consisting of a Distance object and its sampling.

    Attributes
    ----------
    size: int
        integer describing the size of the distance space.
    content: string, ndarray
        either string "empty" or 1D ndarray of non-negative extended real numbers describing the condense form of
        the distance space. The content is "empty" if and only if the attribute size is 0.
    limit: Real
        extended real number used as follows. Every inf entry in content is converted to the biggest finite entry times
        the limit. This is relevant only if the content is ndarray.
    sample: ndarray, str
        can be either:
            - string "all", or
            - string "empty", or
            - 2D ndarray of integers.
    sample_size: int
        integer which is:
            - equal to the attribute size in case the Sample attribute is the string "all".
            - 0 if the Sample attribute is the string "empty".
            - Sample.sample_size in case the case the attribute Sample is a Sample object.
    number_instances: int
        integer which is:
            - 0 if the Sample attribute is the string "empty",
            - 1 if the Sample attribute is the string "all",
            - Sample.number_instances in case this attribute Sample is a Sample object.
     """
    def __init__(self, d, s=None, limit=2):
        """
        Parameters
        ----------
        d: Distance, list[Real], list[list[Real]], ndarray, str
            either:
            -   1D  array like of non-negative extended real numbers, or
            -   a square array like of non-negative extended real numbers with 0 on the diagonal, or
            -   a Distance object, or
            -   string "empty".
        s: None, str, Sample
            either:
                - None, default, in which case the Sample attribute is set to "all", or
                - string "empty", in which case the Sample attribute is set to "empty", or
                - Sample object, in which case the ndarray <code> s.Sample</code> becomes the attribute Sample.
        limit: extended real number
            optional, default 2. This is relevant only when the parameter d is an array like. In this case
            any inf entry in d is converted to the biggest finite entry times the limit parameter.
        """
        if isinstance(d, Distance):
            self.size = d.size
            self.limit = limit
            if d.limit == limit:
                self.content = d.content
            else:
                self.content = Distance(d.content, limit).content
        else:
            self.size = Distance(d, limit).size
            self.content = Distance(d, limit).content
            self.limit = limit
        if s is None:
            self.sample = "all"
            self.sample_size = self.size
            self.number_instances = 1
        elif isinstance(s, str) and s == "empty":
            self.sample = "empty"
            self.sample_size = 0
            self.number_instances = 0
        else:
            self.sample = s.sample
            self.sample_size = s.sample_size
            self.number_instances = s.number_instances

    def get_h0sr(self, clustering_method="single", contour=Contour(Density([[0], [1]]), "dist", inf)):
        """
        Homology stable ranks are the key invariants of distance spaces this software aims at calculating.
        The method get_h0sr retrieves the 0-th homology stable rank of the distance space with respect to the chosen
        hierarchical clustering method and a Contour. Depending on self.Sample a global or averaged stable rank is
        returned:
        -   If self.Sample is "empty": the 0 Pcnif object <code>Pcnif([[0],[0]])</code> is returned.
        -   If self.Sample is "all": a Pcnif object is returned representing the stable rank of the Distance object
            given by self.content. The stable rank is calculated with respect to the specified metric,
            clustering_method, and Contour.
        -   If  self.Sample is an ndarray with at least one row and column: a Pcnif object is returned representing
            the average stable rank constructed as follows:
            -   for each row in self.Sample take the subspace of the distance space restricted to the coordinates
                in the given row.
            -   then calculate the stable rank of this subspace with respect to the chosen clustering method
                and Contour,
            -   finally return the average stable rank across all the indices of self.Sample.

        Parameters
        ----------
        clustering_method: str
            optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", "average", and "ward".
        contour: Contour
            optional, default <code>(Contour(Density([[0], [1]]), "dist", inf)</code>.

        Returns
        -------
        Pcnif object.
        """
        if isinstance(self.sample, str) and self.sample == "empty":
            return Pcnif(np.array([[0], [0]]))
        elif isinstance(self.sample, str) and self.sample == "all":
            return _d_to_h0sr(self.content, clustering_method, contour)
        else:
            d = spatial.distance.squareform(self.content, checks=False)
            return _sd_to_h0sr(d, self.sample, clustering_method, contour)

    def get_bc(self, maxdim=1, thresh=inf, coeff=2):
        """
        Homology stable ranks are the key invariants of distance spaces this software aims at calculating.
        The method get_h0sr retrieves the 0-th homology stable rank with respect to a chosen hierarchical clustering
        method and a Contour. A choice of a hierarchical clustering method is not possible for higher homologies.
        For higher homologies the Vietoris Rips construction is used instead. The ripser software is the computational
        heart behind extracting bar codes of the homology of the Vietoris Rips construction which is the outcome of
        the get_bc method. Depending on self.Sample the following is returned:
        -   If self.Sample is "empty": a dictionary is returned with keys given by
            <code> ["H"+str(d) for d in np.arange(maxdim+1)]</code>. Its values are the empty bar
            codes <code>BC([])</code>.
        -   If self.Sample is "all": a dictionary is returned with keys given by
            <code> ["H"+str(d) for d in np.arange(maxdim+1)]</code>. Its values are the bar-codes of the distance space
            described by self.content.
        -   If self.Sample is an ndarray: a dictinary is returned with keys given by tuples of length 2 of the form
            ("Hd", i) where  $d$ is an integer in the interval $[0,maxdim]$ and $i$ is an integer in the interval
            $[0,self.number_instances]$. Its value for a given key ("Hd", i) is the barcode of the homologies of the
            subspace of the self.content determined by the coordinates of the i-th row in self.Sample.

        Parameters
        ----------
        maxdim: int
            optional, default 1. It is the maximum degree of the calculated homology.
        thresh: Real
            optional, default inf. Distances above this threshold are not considered.
        coeff: int
            optional, default 2. It is a prime number specifying the characteristic of the field of coefficients with
            respect to which the homology is calculated.

        Returns
        ----------
        dictionary
            -   in case self.Sample is "empty" or "all", the keys are given by the list
                <code>["H"+str(d) for d in np.arange(maxdim+1)]</code>.
            -   in case self.Sample is a Sample object, the keys are given by tuples ("Hd", i), where $d$ is an integer
                in the interval $[0,maxdim]$ and $i$ is an integer in the interval  $[0,self.number_instances]$.
        """
        if isinstance(self.sample, str) and self.sample == "empty":
            return _empty_bc(maxdim)
        elif isinstance(self.sample, str) and self.sample == "all":
            return _d_to_bc(self.content, maxdim, thresh, coeff)
        else:
            sf = spatial.distance.squareform(self.content, checks=False)
            out = {}
            inst = 0
            _b = {}
            while inst < self.number_instances:
                _s = self.sample[inst]
                _b[inst] = _d_to_bc(sf[np.ix_(_s, _s)], maxdim, thresh, coeff)
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


def _d_to_h0sr(d, clustering_method, contour, link=None):
    """d is assumed to be a 1D condense distance matrix"""
    if len(d) == 0:
        return Pcnif([[0], [1]])
    else:
        link = linkage(d, clustering_method)
        g = _linkage_to_stable_rank(link, contour)
        return g


def _d_to_bc(d, maxdim, thresh, coeff):
    """d is assumed to be a 2D square ndarray distance matrix with 0 on the diagonal"""
    dgms = ripser(d, maxdim=maxdim, thresh=thresh, coeff=coeff, distance_matrix=True, do_cocycles=False)["dgms"]
    out = {"H" + str(h): BC(dgms[h]) for h in range(maxdim+1)}
    out = {}
    ind = 0
    while ind <= maxdim:
        out["H" + str(ind)] = BC(dgms[ind])
        ind += 1
    return out


def _sd_to_h0sr(d, s, clustering_method, contour):
    """d is assumed to be a 2D square ndarray distance matrix with 0 on the diagonal and s 2D ndarray
    representing a sample"""
    f = Pcnif([[0.], [0.]])
    _i = 0
    number_instances = len(s)
    while _i < number_instances:
        samp = s[_i]
        r_d = spatial.distance.squareform(d[np.ix_(samp, samp)], checks=False)
        g = _d_to_h0sr(r_d, clustering_method, contour)
        f += g
        _i += 1
    return f * (1 / number_instances)


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


def _interval_subdivide(b, e, n, overlap):
    a = np.zeros([n, 2])
    s = (e - b) / n
    a[0] = [b, b + s + (s * overlap) / 100]
    _i = 1
    while _i < n - 1:
        a[_i] = [b + s*_i - (s*overlap)/100, b + s*(_i + 1) + (s*overlap)/100]
        _i += 1
    a[n - 1] = [b + (n - 1)*s - (s*overlap)/100, e]
    return a


def _filter_subdivide(f, number_intervals, overlap, b, e):
    intervals = _interval_subdivide(b, e, number_intervals, overlap)
    cov = {}
    _i = 0
    while _i < len(intervals):
        start = intervals[_i][0]
        end = intervals[_i][1]
        cov[_i] = tuple(np.where(np.logical_and(f >= start, f <= end))[0])
        _i += 1
    return Cover(cover=cov)


def _dict_dict_cover(cover):
    """cover is assumed to be a dictionary of dictionaries"""
    cov = {}
    for _i in cover.keys():
        for _j in cover[_i].keys():
            cov[(_i, _j)] = cover[_i][_j]
    return Cover(cover=cov)


