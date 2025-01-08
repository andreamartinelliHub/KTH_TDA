#!/usr/bin/env python3
"""
Created on Sun June 13 18:27 2021

@author: wojtek
"""

import numpy as np
import matplotlib.pyplot as plt

from stablerank.rtorf import Pcf
from stablerank.rtorf import Density
from stablerank.rtorf import Pcnif
from stablerank.rtorf import Contour

inf = float("inf")


class BC(object):
    r"""
    A bar code is a pair of non-negative extended real numbers $[a,b)$ such that $a< b$. Objects in this
    class represent lists of bar codes.

    Attributes
    ----------
    bars: ndarray
        nx2 ndarray whose rows represent bars.
    """
    def __init__(self, bars):
        """
        Parameters
        ----------
        bars: list[Real], list[list[Real]], ndarray
            can be either:
            -   1D array like of length 2 of non negative extended real numbers which represents a bar
                (its first coordinate is strictly smaller that the second), or
            -   nx2 (2D) array like of extended real numbers whose rows represent bars
                (their first coordinates are strictly smaller that their second coordinates).
        Raises
        ------
        ValueError
            If the input is not 1D array like of length 2 or nx2 2D array like.
        """
        if np.ndim(bars) == 1 and len(bars) == 2:
            self.bars = np.asarray([bars])
        elif np.ndim(bars) == 2 and np.shape(bars)[1] == 2:
            self.bars = np.asarray(bars)
        else:
            raise ValueError("""The input should be either 1D array like of length 2 or nx2 2D array like""")

    def length(self, c=Contour(Density([[0], [1]]), "dist", inf)):
        """
        Produces a list of length of the bars in the BC object. Their length is calculated with respect to the specified
        contour.

        Parameters
        ----------
        c: Contour
            contour object, optional, default <code>contour(Density([[0], [1]]), "dist", inf)</code>,
            which is an essential ingredient in calculating the length of bars.

        Returns
        -------
        1D ndarray
            of non-negative extended real numbers describing the length of the bars with respect to the specified
            contour.
        """
        if len(self.bars) == 0:
            return np.empty([0])
        else:
            def g(x): return c.density.bar_length(x, c.kind, c.truncation)
            outcome = np.apply_along_axis(g, 1, self.bars)
            return outcome

    def stable_rank(self, c=Contour(Density([[0], [1]]), "dist", inf), p=inf, q=1):
        r"""
        Converts a BC object into Pcnif object, called stable rank, with resect to the chosen contour.
        Stable ranks are the key invariants this software aims at calculating and analysing.

        Parameters
        ----------
        c: Contour
            contour object, optional, default <code>contour(Density([[0], [1]]), "dist", inf)</code>,
            which is an essential ingredient in calculating the length of bars.
        p: Algebraic Wasserstein "p" parameter
            Real extended number in $[1, \infty]$
        q: Algebraic Wasserstein "q" parameter
            Real extended number in $[1, \infty]$

        Returns
        -------
        Pcnif object.
        """
        if len(self.bars) == 0:
            return Pcnif(np.array([[0], [0]]))

        length = self.length(c)

        if p == inf:
            sort_length = np.unique(length, return_counts=True)
            dom = sort_length[0]
            values = np.cumsum(sort_length[1][::-1])[::-1]
        else:
            length.sort()
            dom = np.cumsum(length ** p) ** (1 / p)
            values = np.arange(1, len(length) + 1)[::-1]

        if q == inf:
            dom *= (1 / 2)
        else:
            dom *= 2 ** ((1 - q) / q)

        n_inf = sum(dom == inf)
        if n_inf > 0:
            dom = np.insert(dom[:-n_inf], 0, 0)
            if n_inf > 1:
                values = values[:-(n_inf - 1)]
        else:
            dom = np.insert(dom, 0, 0)
            values = np.append(values, 0)

        return Pcnif(np.vstack((dom, values)))


    def plot(self):
        """
        Plots the bar code. In the plot the blue lines represent the bars that have inf as their ends, and the red
            lines represent finite bars.

        Returns
        -------
        pyplot
            of the bars in the BC object.
        """
        plt.yticks([])
        bars = self.bars
        number_bars = len(bars)
        if number_bars > 0:
            m = np.amax(bars[bars != np.inf])
            ind_fin = np.isfinite(bars).all(axis=1)
            bars_fin = bars[ind_fin]
            pos_fin = np.arange(0, len(bars_fin))
            plt.hlines(pos_fin, bars_fin[:, 0], bars_fin[:, 1],
                       colors=["red"], linewidth=0.6)
            ind_inf = np.isinf(bars).any(axis=1)
            bars_inf = bars[ind_inf]
            pos_inf = np.arange(len(bars_fin),
                                len(bars_fin) + len(bars_inf))
            if m == 0:
                ends = np.ones(len(bars_inf))*10
            else:
                ends = np.ones(len(bars_inf)) * 2 * m
            plt.hlines(pos_inf, bars_inf[:, 0], ends,
                       colors=["blue"], linewidth=0.6)
        else:
            plt.text(0.2, 0.2, "empty bar code")

    def betti(self):
        b = self.bars
        dom = np.sort(np.reshape(b, -1))
        val = np.empty(len(dom))
        j = 0
        while j < len(dom):
            x = dom[j]
            v = 0
            i = 0
            while i < len(b):
                if b[i][0] <= x < b[i][1]:
                    v = v + 1
                i = i + 1
            val[j] = v
            j = j + 1
        return Pcf(np.vstack([dom, val]))

def empty_bc(maxdim):
    out = {"H" + str(h): BC(np.empty([0, 2])) for h in range(maxdim+1)}
    return out


def one_point_bc(maxdim):
    out = {"H0": BC([[0, inf]])}
    h = 1
    while h <= maxdim:
        out["H" + str(h)] = BC(np.empty([0, 2]))
        h += 1
    return out

