#!/usr/bin/env python3
"""
Created on Thu June 10 22:27 2021

@author: wojtek
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


class Cover(object):
    r"""
    Objects in this class represent covers, which describe a set as a union of its family of subsets. Assume a set is
    covered by a family of its subsets: $$X=\bigcup_{i\in I} X_i$$
    -   Elements of $X$ are called **observations**.
    -   The subsets $X_i$ are called  **blocks**.
    -   $I$ is called the **index**.
    -   The dictionary $\\{i: X_i\\}_{i\in I}$ is called the **cover**
    -   For any observation $x$ in $X$, the tuple $(i\in I\ |\ x\in X_i)$ is called the **container**
    containing $x$. The tuple $(i\in I\ |\ X_i=\emptyset)$ is called the **container** of empty blocks. Containers are
    assembled into the following dictionary called **containers**:
    $$x: (i\in I\ |\ x\in X_i)$$
    $$\text{"empty_blocks"}: (i\in I\ |\ X_i=\emptyset)
    $$

    Attributes
    ----------
    cover: Dictionary
        the dictionary $\\{i: X_i\\}_{i\in I}$. Its values are tuples.
    index: tuple
        the tuple given by keys of the dictionary cover.
    observations: tuple
        the tuple given by the elements of the set $X$.
    containers: Dictionary
        the dictionary whose keys are observations plus an additional key called "empty_blocks".
        Its values are the corresponding containers.
    index_enum: Dictionary
        the dictionary whose keys are the elements in the tuple index and whose value at the given key is the place
        of this element in the tuple index.
    observations_enum: Dictionary
        the Dictionary whose keys are the elements in the tuple observations and whose value at the given key is
        the place of this element in the tuple observations.
    """
    def __init__(self, cover=None, containers=None, partition=None, partition_labels=None):
        """
        Parameters
        ----------
        cover: None, Dictionary, Series
            optional, default None. If it is not None, then it has to be either a dictionary or a panda Series whose
            values are assumed to be lists or tuples or 1D array like which might be empty. These values can not
            contain the string "empty_blocks" and have to be hashable objects. The keys/index of this parameter
            have to be also hashable. Recall that a string, integer, real number, tuple is hashable, however a list
            is not. In this case the following attributes are assigned to the object:
            -   *index*: is the tuple given by the keys/indexes of the cover,
            -   *observations*: is the *tuple* of all the elements appearing in the lists/tuples/1D arrays
                which are values of the parameter cover,
            -   *cover*: is the dictionary obtained from the parameter cover by converting all its values into tuples,
            -   *containers*: is the dictionary whose keys are the observations plus additional key given by the string
                "empty_blocks". The values of this attribute are tuples given by the corresponding containers.
                The container corresponding to an observation is the tuple of all the keys i such that
                <code>cover[i]</code> contains the observation. The container corresponding to the key
                "empty_blocks" is the tuple of all the indexes i such that <code>cover[i]</code> is empty.
            -   *index_enum*: is the dictionary whose keys are the elements in the attribute *index* and whose value at
                the given key is the place of this element in the tuple *index*.
            -   *observations_enum*: is the dictionary whose keys are the elements in the attribute *observations* and
                whose value at the given key is the place of this element in the tuple *observations*.

            If the parameter cover is None, then the next parameter **containers** is checked:

        containers: None, Dictionary, Series
            optional, default None. If it is not None, then it has to be either a dictionary or a panda Series whose
            values are assumed to be lists or tuples or 1D array like which might be empty. These values have to
            be hashable objects. In the case there is "empty_blocks" in the keys/index of the parameter containers, then
            the associated list has to be disjoint from all the other lists which are values of this parameter.
            In this case the following attributes are assigned to the object:
            -   *index*: is the tuple of all the elements appearing in the lists/tuples/1D arrays which are values of
                the panda Series containers,
            -   *observations*: is the tuple given by the indexes of the panda Series containers which are different
                from "empty_blocks" and whose values are non empty lists/tuples/1D arrays,
            -   *cover*: is the dictionary whose keys are given by index and whose value at a given index is the tuple
                of all the observations (keys of the parameter containers) for which the value of the corresponding
                container contains the given index.
            -   *containers*: is the dictionary whose keys obtained from the parameter containers by
                converting all its values into tuples. In the case the string "empty_blocks" is not in the keys/index of
                the parameter containers, then this key is added to the attribute *containers* and the value assigned
                to this index is the empty tuple.
            -   *index_enum*: is the dictionary whose keys are the elements in the attribute *index* and whose value
                at the given key is the place of this element in the tuple *index*.
            -   *observations_enum*: is the dictionary whose keys are the elements in the tuple *observations* and whose
                value at the given key is the place of this element in the attribute *observations*.

            If the parameters cover and containers are None, then the next parameters **partition** and **labels**
            are checked:

        partition: None, tuple, list, ndarray
            optional, default None. If not None, then is has to be a lists or tuple or 1D array like whose values have
            to be hashable objects. In this case the following attributes are assigned to the object:
            -   *observations*:
                - if labels is None, then *observations* is the tuple consisting of all the integers between 0 and
                <code>len(partition)</code>.
                - if labels is 1D array like, then *observations* is the tuple <code>tuple(labels)</code>
            -   *index*: is the tuple consisting of all the values of the parameter partition.
            -   *cover*: is the dictionary whose keys are given by the attribute index, which are values of the
                partition, and assigning to such a value the following tuple:
                - if labels is None, then the tuple consists of all the coordinates of the parameter partition with
                the given value.
                - if labels is 1D array like, then the tuple consists of the values of labels at coordinates where
                the parameter partition has the given value.
            -   *containers*: is the dictionary whose keys are given by the attribute observations plus an extra key
                given by the string "empty_blocks". Its value at a given observation is the one element tuple consisting
                of the value of the array partition at the specified observation. The value at the index "empty_blocks"
                is the empty tuple.
            -   *index_enum*: is the dictionary whose keys the elements in the tuple *index* and whose
                value at the given key is the place of this element in the attribute *index*.
            -   *observations_enum*: is the dictionary whose keys are the elements in the tuple *observations* and whose
                value at the given key is the place of this element in the attribute *observations*.
        partition_labels: None, tuple, list, ndarray
            optional, default None. If not None, then it has to be a list like of differant elements that are
            hashable and can contain the string "empty_blocks".  Elements in the parameter labels are the names of the observations.
            If not None, then the size of the parameter labels should be the same as the size of the parameter
            partition.
        """
        if isinstance(cover, dict):
            info = _dict_cover_to_cover(cover)
        elif isinstance(cover, pd.core.series.Series):
            info = _series_cover_to_cover(cover)
        elif isinstance(containers, dict):
            info = _dict_containers_to_cover(containers)
        elif isinstance(containers, pd.core.series.Series):
            info = _series_containers_to_cover(containers)
        elif isinstance(partition, (list, tuple, np.ndarray)):
            info = _partition_to_cover(partition, partition_labels)
        else:
            raise ValueError(""""Either Cover or containers or partition needs to be specified""")
        self.cover = info["cover"]
        self.observations = info["observations"]
        self.index = info["index"]
        self.containers = info["containers"]
        self.index_enum = {}
        _i = 0
        while _i < len(self.index):
            self.index_enum[self.index[_i]] = _i
            _i += 1
        self.observations_enum = {}
        _i = 0
        while _i < len(self.observations):
            self.observations_enum[self.observations[_i]] = _i
            _i += 1

    def cover_size(self):
        r"""
        If self represents $\\{X_i\\}_{i\in I}$, then this method returns the dictionary
        $\\{i: |X_i|\\}_{i\in I}$.

        Returns
        ----------
        dictionary
        """
        out = {}
        for _i in self.index:
            out[_i] = len(self.cover[_i])
        return out

    def container_size(self):
        r"""
        If self represents $\\{X_i\\}_{i\in I}$, then this method returns the dictionary
        $\\{x: |\\{i: x\in X_i\\}|\\}_{x\in X}$.

        Returns
        ----------
        dictionary
        """
        out = {}
        for _ob in self.observations:
            out[_ob] = len(self.containers[_ob])
        return out

    def distance(self, metric="relative_Jacquard"):
        r"""
        It produces a 1D ndarray of real numbers which is the condensed form of distances between the blocks of
        the cover with respect to the specified metric.
        Parameters
        ----------
        metric: str
            optional, default "relative_Jacquard". It specifies which metric is used to calculate the distances between
            the blocks of the cover. Possible choices are "symmetric", "Jacquard", and "relative_Jacquard". Recall that
            -   symmetric metric: $d(X_i,X_j)=|X_i\setminus X_j|+|X_j\setminus X_i|$.
            -   Jacquard metric: $d(X_i,X_j)=1-\frac{|X_i\cap X_j|}{|X_i\cup X_i|}$ if $X_i\cup X_j\not =\emptyset$.
            -   relative_Jacquard metric: $d(X_i,X_j)=1-\frac{|X_i\cap X_j|}{\text{min}(|X_i|,\  |X_i|)}$ if
                $X_i\not =\emptyset$ and $X_j\not =\emptyset$.

        Returns
        ----------
        ndarray
            1D ndarray which is the condensed form of distances between the blocks of the cover.
        """
        size = len(self.index)
        d = np.array([])
        _i = 0
        if metric == "symmetric":
            while _i < size:
                block_i = set(self.cover[self.index[_i]])
                _j = _i + 1
                while _j < size:
                    block_j = set(self.cover[self.index[_j]])
                    d = np.append(d, len(block_i.symmetric_difference(block_j)))
                    _j += 1
                _i += 1
            return d
        elif metric == "Jacquard":
            while _i < size:
                cov_i = set(self.cover[self.index[_i]])
                _j = _i + 1
                while _j < size:
                    cov_j = set(self.cover[self.index[_j]])
                    enum = len(cov_i.intersection(cov_j))
                    denum = len(cov_i.union(cov_j))
                    if denum == 0:
                        d = np.append(d, 1)
                    else:
                        d = np.append(d, 1 - enum / denum)
                    _j += 1
                _i += 1
            return d
        elif metric == "relative_Jacquard":
            while _i < size:
                cov_i = set(self.cover[self.index[_i]])
                _j = _i + 1
                while _j < size:
                    cov_j = set(self.cover[self.index[_j]])
                    enum = len(cov_i.intersection(cov_j))
                    denum = min(len(cov_i), len(cov_j))
                    if denum == 0:
                        d = np.append(d, 1)
                    else:
                        d = np.append(d, 1 - enum / denum)
                    _j += 1
                _i += 1
            return d
        else:
            raise ValueError("""You need to choose the metric among "symmetric","Jacquard" or "relative_Jacquard" """)

    def stack(self, other):
        r"""
        In the case other is a cover then it is assumed (but not checked) that the index of the self object coincides
        with the observations of the other object. In the case the other is 1D array like, then it is assumed
        (but not checked) that its length is the size of the index of the self object. In this case we think about
        this 1D array like as partition of the indexes of the self object.

        If $X =\cup_{i\in I}X_i$ and $I = \cup_{j\in J}Y_j$  are represented by respectively the self and the other
        objects, then  this method returns the cover object which represents the following cover:
        $$X= \cup_{j\in J} (\cup_{i\in J_j }X_i)$$

        Parameters
        ----------
        other: Cover, list, tuple, ndarray
            the cover that is stuck on the self cover

        Returns
        ----------
        Cover
        """
        if isinstance(other, Cover):
            o_cov = other.cover
            o_ind = other.index
        else:
            containers = {}
            for _i in self.index:
                containers[_i] = [other[self.index_enum[_i]]]
            out = _dict_containers_to_cover(containers)
            o_cov = out["cover"]
            o_ind = out["index"]
        cov = {}
        for j in o_ind:
            block = []
            for x in o_cov[j]:
                block += list(self.cover[x])
            cov[j] = tuple(set(block))
        return Cover(cover=cov)

    def contract(self, metric="relative_Jacquard", clustering_method="single",
                 contraction_rate=None, number_cluster=None, cut_off=0.7):
        """
        This method contracts a Cover object to a new Cover object with a possibly smaller number of blocks.
        The contraction is done as follows:
        -   distances between the blocks are calculated with respect to the chosen metric ("symmetric", "Jacquard", or
        "relative_Jacquard"). In this way we get distances between the indexes of the Cover object.
        -   a linkage of the obtained distance space (on the indexes of the Cover object) is formed with respect to
        the chosen clustering method ("single", "complete", "average").
        -   if the parameter contraction_rate is specified, then the obtained linkage is converted into a partition
        of the index of the Cover object whose number of clusters does not exceed contraction_rate times
        the size of the index of the Cover object. If this parameter is not specified, then the parameter
        number_cluster is checked.
        -   if the parameter number_cluster is specified, then the obtained linkage is converted into a partition
        of the index of the Cover object whose number of clusters does not exceed the specified number of
        the size of the index of the Cover object. If this parameter is not specified, then the parameter
        cut_off is considered.
        -   the obtained linkage is converted into a partition of the index of the Cover object so that the original
        indexes in each  cluster have no greater a cophenetic distance than cut_off.
        -   the contraction is obtained by stacking the Cover object with the obtained partition.

        Parameters
        ----------
        metric: str
            optional, default "relative_Jacquard". It specifies which metric is used to calculate the distances between
            the blocks of the cover. Possible choices are "symmetric", "Jacquard", and "relative_Jacquard".
        clustering_method: str
            optional, default "single". It is a string specifying which hierarchical clustering scheme is used.
            Among possible choices are "single", "complete", and "average".
        contraction_rate: None, Real
            optional, default None. It should be positive but not bigger than 1. It specifies the contraction rate.
        number_cluster: None, Int
            optional, default None. It should be a positive integer. It specifies the upper limit of how many blocks
            the contracted cover should have.
        cut_off: Real
            optional, default 0.7. It specifies the cophenetic distance cut-off for the dendrogram.

        Returns
        ----------
        Cover
        """
        d = self.distance(metric)
        link = linkage(d, clustering_method)
        size = len(self.index)
        if contraction_rate is None:
            if number_cluster is None:
                a = fcluster(link, cut_off, criterion='distance')
                return self.stack(a)
            else:
                a = fcluster(link, number_cluster, criterion='maxclust')
                return self.stack(a)
        else:
            mc = int(size * contraction_rate / 100)
            a = fcluster(link, mc, criterion='maxclust')
            return self.stack(a)

    def __mul__(self, other):
        r"""
        It is assumed that self and other represent covers  $\\{X_i\\}_{i\in I}$ and  $\\{Y_j\\}_{j\in J}$ of the same
        set of observations. In which case a cover object representing $\\{X_i\cap Y_j\\}_{(i,j)\in I\times J}$
        is returned.

        Parameters
        ----------
        other: Cover
            it is a Cover object assumed to have the same observations as the object self.

        Returns
        ----------
        Cover
        """
        out_c = {}
        for _i in self.index:
            s1 = set(self.cover[_i])
            for _j in other.index:
                s2 = set(other.cover[_j])
                out_c[(_i, _j)] = tuple(s1.intersection(s2))
        return Cover(cover=out_c)

    def proportion(self, other):
        r"""
        It is assumed that self and other represent covers  $\\{X_i\\}_{i\in I}$ and  $\\{Y_j\\}_{j\in J}$ of the same
        set of observations. In which case a dictionary with keys given by the tuples in the product $I\times J$ is
        returned. Its value at the key $(i,j)$ is either the real number 0, if $Y_j$ is empty, and
        $|X_i\cap Y_j|/|Y_j|$, if $Y_j$ is not empty.
        """
        out_p = {}
        for _i in self.index:
            s1 = set(self.cover[_i])
            for _j in other.index:
                s2 = set(other.cover[_j])
                _d = len(s2)
                if _d == 0:
                    out_p[(_i, _j)] = 0
                else:
                    _e = len(s1.intersection(s2))
                    out_p[(_i, _j)] = _e / _d
        return out_p


def cover_product(list_covers):
    r"""
    The parameter list_covers  is assumed to be a list of covers of a given set of observations $X$:
    $$\[X=\bigcup_{i\in I_k}X_{i_k}^k\ |\ 0\leq k < n\]$$
    This method returns a cover representing:
    $$ X=\bigcup_{(i_0,\ldots, i_{n-1})\in I_0\times\cdots I_{n-1}} X^0_{i_0}\cap\cdots\cap  X^{n-1}_{i_{n-1}} $$


    Parameters
    ----------
    list_covers: list
        it is assumed to be a non empty list of Cover objects

    Returns
    ----------
    Cover
    """
    _l = len(list_covers)
    if _l == 1:
        return list_covers[0]
    else:
        cov = {}
        for _k in list_covers[0].index:
            cov[tuple(_k)] = set(list_covers[0].cover[_k])
        _i = 1
        while _i < _l:
            out = {}
            for _k in cov.keys():
                print(_k)
                for _ind in list_covers[_i].index:
                    print(_ind)
                    s2 = set(list_covers[_i].cover[_ind])
                    _w = list(_k)
                    _w.append(_ind)
                    out[tuple(_w)] = cov[_k].intersection(s2)
            cov = out
            _i += 1
        return Cover(cover=out)


def _series_cover_to_cover(cover):
    """cover is assumed to be a panda Series"""
    cov = {}
    ob = []
    empty = []
    for _i in cover.index:
        _t = tuple(cover.at[_i])
        cov[_i] = _t
        ob.extend(list(cover.at[_i]))
        if len(_t) == 0:
            empty.append(_i)
    ob = tuple(set(ob))
    ind = tuple(set(cover.index))
    cont = {}
    for o in ob:
        c = []
        for b in ind:
            if o in cover.at[b]:
                c.append(b)
        cont[o] = tuple(c)
    cont["empty_blocks"] = tuple(empty)
    return {"cover": cov, "observations": ob, "index": ind, "containers": cont}


def _dict_cover_to_cover(cover):
    """cover is assumed to be a dictionary"""
    cov = {}
    ob = []
    empty = []
    for _i in cover.keys():
        _t = tuple(cover[_i])
        cov[_i] = _t
        ob.extend(list(cover[_i]))
        if len(_t) == 0:
            empty.append(_i)
    ob = tuple(set(ob))
    ind = tuple(set(cover.keys()))
    cont = {}
    for o in ob:
        c = []
        for b in ind:
            if o in cover[b]:
                c.append(b)
        cont[o] = tuple(c)
    cont["empty_blocks"] = tuple(empty)
    return {"cover": cov, "observations": ob, "index": ind, "containers": cont}


def _series_containers_to_cover(containers):
    """containers is assumed to be a panda Series"""
    ob = tuple([x for x in containers.index if x != "empty_blocks" and len(containers.at[x]) > 0])
    cont = {}
    cov = {}
    ind = []
    for x in ob:
        ind.extend(list(containers.at[x]))
        cont[x] = tuple(containers.at[x])
    if "empty_blocks" in containers.index:
        ind.extend(list(containers.at["empty_blocks"]))
        cont["empty_blocks"] = containers.at["empty_blocks"]
        for _i in containers.at["empty_blocks"]:
            cov[_i] = tuple([])
    else:
        cont["empty_blocks"] = tuple([])
    ind = tuple(set(ind))
    for _i in ind:
        block = []
        for x in ob:
            if _i in containers.at[x]:
                block.append(x)
        cov[_i] = tuple(block)
    return {"cover": cov, "observations": ob, "index": ind, "containers": cont}


def _dict_containers_to_cover(containers):
    """containers is assumed to be a dictionary"""
    ob = tuple([x for x in containers.keys() if x != "empty_blocks" and len(containers[x]) > 0])
    cont = {}
    cov = {}
    ind = []
    for x in ob:
        ind.extend(list(containers[x]))
        cont[x] = tuple(containers[x])
    if "empty_blocks" in containers.keys():
        ind.extend(list(containers["empty_blocks"]))
        cont["empty_blocks"] = containers["empty_blocks"]
        for _i in containers["empty_blocks"]:
            cov[_i] = tuple([])
    else:
        cont["empty_blocks"] = tuple([])
    ind = tuple(set(ind))
    for _i in ind:
        block = []
        for x in ob:
            if _i in containers[x]:
                block.append(x)
        cov[_i] = tuple(block)
    return {"cover": cov, "observations": ob, "index": ind, "containers": cont}


def _partition_to_cover(partition, labels=None):
    if labels is None:
        labels = range(len(partition))
    cont = {}
    _i = 0
    for _l in labels:
        cont[_l] = tuple([partition[_i]])
        _i += 1
    return _dict_containers_to_cover(cont)


