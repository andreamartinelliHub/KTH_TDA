#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:22:46 2021

@author: wojtek
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

inf = float("inf")


class Pcf(object):
    r"""
    Objects in this class represent piecewise constant functions $f\colon (-\infty,\infty)\to (-\infty,\infty)$. To
    define a Pcf, we need to specify an array with two rows. The 0-th row describes the domain and should consist of
    strictly increasing real numbers. The 1-st row describes the values of the function. For example a Pcf encoded by
    the following array is a function which has value $0$ on the interval $(-\infty,1)$, has value $-1$ on the interval
    $[1,2)$, and has value $3$ on the interval $[2,\infty)$: $$\begin{bmatrix}1 & 2\\\\ -1 & 3 \end{bmatrix}$$

    The conditions assumed about the 0-th row are not checked!

    Attributes
    ----------
    content: ndarray
        2xn ndarray of real numbers. Its 0-th row is the domain and the 1-st row consists of the values of the Pcf.
    """
        
    def __init__(self, a):
        """
        Parameters
        ----------
        a: list[list[Real]], ndarray
            2xn array like of real numbers. It is assume that $n>0$ and that the 0-th row, referred to as the domain,
            consists of strictly increasing real numbers. The second row consists of values and any real value is
            allowed.
        """
        self.content = np.asarray(a, dtype=float)

    def evaluate(self, x):
        """
        Evaluates the pcf at given values.

        Parameters
        ----------
        x: Real, list[Real], ndarray
            a real number or an array like of real numbers. These are the real numbers at which the pcf specified
            by self is evaluated at.

        Returns
        ------
        real number
            which is the value of the pcf specified by self at x,  in case x is a real number, or
        ndarray
            of real numbers which are the values of this pcf at coordinates of x, in case x is array like.
        """
        v = np.concatenate(([0], self.content[1]))
        x_place = np.searchsorted(self.content[0], x, side='right')
        return v[x_place]

    def restrict(self, start, end):
        """
        Produces a 2xn ndarray of the restriction of the pcf specified by self to the interval $[start,end]$.

        Parameters
        ----------
        start: Real
            an extended real number. The start of the interval the pcf specified by self is restricted to.
        end: Real
            extended real number. The end of the interval the pcf is restricted to.
            It has to be strictly bigger than the start.

        Returns
        ------
        ndarray of real numbers
            of dimension 2xn. This array represents the restriction of the pcf specified by self to the
            interval $[start,end]$.
        """
        s_place = np.searchsorted(self.content[0], start, side='right')
        if s_place > 0 and end < inf:
            e_place = np.searchsorted(self.content[0], end)
            v = self.content[:, s_place:e_place]
            return np.concatenate(([[start], [self.evaluate(start)]], v, [[end], [0]]), axis=1)
        elif s_place == 0 and end < inf:
            e_place = np.searchsorted(self.content[0], end)
            v = self.content[:, s_place:e_place]
            return np.concatenate((v, [[end], [0]]), axis=1)
        elif s_place > 0 and end == inf:
            v = self.content[:, s_place:]
            return np.concatenate(([[start], [self.evaluate(start)]], v), axis=1)
        else:
            return self.content

    def _extend(self, d):
        """
        Produces a 2xn ndarray where the 0th row is an extension of the domain by the given vector. The 1st row
        describes the values.

        Parameters
        ----------
        d: list[Real] or ndarray
            1D array like of real numbers. These are the elements added to the domain.

        Returns
        ------
        ndarray of real numbers
            of dimension 2xm. The 0-th row of this array is the extended domain.
        """
        dom = self.content[0]
        domain = np.unique(np.concatenate([dom, d]))
        parameters = np.searchsorted(dom, domain, side='right')-1
        val = np.append(self.content[1], 0)[parameters]
        return np.vstack((domain, val))

    def simplify(self):
        """
        If some values of self.content are repeated consecutively, then all these values except the first one are
        omitted. The outcome is an 2xn ndarray.

        Returns
        ------
        ndarray of real numbers
            of dimension 2xn. The 1-st row does not contain any consecutively repeated values.
        """
        c = self.content[1][:-1]-self.content[1][1:]
        k = np.insert(np.where(c != 0)[0] + 1, 0, 0)
        return self.content[:, k]

    def plot(self, start=-inf, end=inf, ext_l=0, ext_r=0.1, ax=None, **kwargs):
        """
        Plots the pcf.

        Parameters
        ----------
        start: Real
            extended real number, optional, default -inf. It specifies the start point for the plot's domain.
        end: Real
            extended real number, optional, default inf.  It specifies the end points for the plot's domain.
        ext_l: Real
            a real number,  optional,  default 0. It describes how much to the left the plot's domain is extended.
        ext_r: Real
            a real number,  optional,  default 0.1. It describes how much to the right the plot's domain is extended.
        ax: None, pyplot axes
            optional, default None. In case it is None  plt.plot is produced. Otherwise a plt.axes is
            produced.
        **kwargs: additional parameters
            the keyword arguments for the returned axes class, which include color, linewidth.

        Returns
        -------
        pyplot plot or pyplot axes which contain the plot of the Pcf.
        """
        b = self.restrict(start, end)
        if ext_l == 0 and ext_r > 0:
            c = [[b[0, -1] + ext_r], [b[1, -1]]]
            b = np.concatenate((b, c), axis=1)
        elif ext_l > 0 and ext_r == 0:
            a = [[b[0, 0] - ext_l], [0]]
            b = np.concatenate((a, b), axis=1)
        else:
            a = [[b[0, 0] - ext_l],
                 [0]]
            c = [[b[0, -1] + ext_r], [b[1, -1]]]
            b = np.concatenate((a, b, c), axis=1)
        if ax is None:
            ax = plt
        return ax.step(b[0], b[1], where='post', **kwargs)

    def __add__(self, other):
        """
        Adds a pcf or a real number to the pcf specified by self. We can use both
        <code>f + g</code> or <code>f.add(g)</code>.

        Parameters
        ----------
        other: Real, Pcf
            either a pcf or real number which is added to the pcf specified by self.

        Returns
        ------
        Pcf
            which is the result of the addition

        Raises
        ------
        ValueError
            We can only add to a pcf another pcf or a real number
        """
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            c = np.vstack((f[0], f[1] + g[1]))
            if isinstance(self, Pcnif) and isinstance(other, Pcnif):
                return Pcnif(c)
            if isinstance(self, Density) and isinstance(other, Density):
                return Density(c)
            return Pcf(c)
        if isinstance(other, (int, float)):
            c = self.content + np.array([[0], [other]], dtype='float64')
            if isinstance(self, Pcnif) and other >= 0.:
                return Pcnif(c)
            if isinstance(self, Density) and c > 0.:
                return Density(c)
            return Pcf(c)
        raise ValueError("""we can only add to a pcf a pcf or a real number""")

    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        """
        Produces the negative of the pcf specified by self. We can use both <code>- f</code> or <code>f.neg()</code>.

        Returns
        ------
        Pcf
            which is the negative of the pcf specified by self.
        """
        return Pcf(np.vstack((self.content[0], -self.content[1])))

    def __mul__(self, other):
        """
        Multiplies the pcf specified by self by a pcf or a real number. We can use both <code>f * g</code> or
        <code>f.mul(g)</code>.

        Parameters
        ----------
        other: Real, Pcf
            either a pcf or real number, which is what the pcf specified by self is multiplied by.

        Returns
        ------
        Pcf
            which is the result of the multiplication.

        Raises
        ------
        ValueError
            we can only multiply pcf by a pcf or a real number
        """
        if isinstance(other, (int, float)):
            c = self.content * np.array([[1], [other]], dtype='float64')
            if other > 0:
                if isinstance(self, Pcnif):
                    return Pcnif(c)
                elif isinstance(self, Density):
                    return Density(c)
                else:
                    return Pcf(c)
            if other < 0:
                return Pcf(c)
            if isinstance(self, Pcnif):
                return Pcnif([[0], [0]])
            return Pcf([[0], [0]])
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            c = np.vstack((f[0], f[1] * g[1]))
            if isinstance(self, Pcnif) and isinstance(other, Pcnif):
                return Pcnif(c)
            if isinstance(self, Density) and isinstance(other, Density):
                return Density(c)
            return Pcf(c)
        raise ValueError("""we can only multiply a pcf by a pcf or a real number""")
       
    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        """
        Subtracts a pcf or a real number from the pcf specified by self. We can use both <code>f - g</code> or
        <code>f.sub(g)</code>.

        Parameters
        ----------
        other: Real, Pcf
            either a pcf or real number, which is what is subtracted from the pcf specified by self.

        Returns
        ------
        Pcf
            which is the result of the subtraction.

        Raises
        ------
        ValueError
            we can only subtract from a pcf a pcf or a real number
        """
        if isinstance(other, (int, float)):
            c = self.content - np.array([[0], [other]], dtype='float64')
            return Pcf(c)
        if isinstance(other, Pcf):
            f = self._extend(other.content[0])
            g = other._extend(self.content[0])
            c = np.vstack((f[0], f[1] - g[1]))
            return Pcf(c)
        raise ValueError("""we can only subtract from a pcf a pcf or a real number""")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Pcf([[0], [other]]) - self
        raise ValueError("""we can subtract a pcf only from a real number or a  pcf""")

    def __pow__(self, p):
        """
        Raises the pcf specified by self to the specified power. Infinite values can appear in case the power is
        negative and some value is 0. We can use both <code>f ** p</code> or <code>f.pow(p)</code>.

        Parameters
        ----------
        p: Real
            real number. It is the power to which the Pcf is raised to.

        Returns
        -------
        Pcf
            which is the result of raising the given pcf to the p-th power.
        """
        c = np.vstack((self.content[0], self.content[1] ** p))
        if isinstance(self, Pcnif) and p >= 1:
            return Pcnif(c)
        elif isinstance(self, Density):
            return Density(c)
        else:
            return Pcf(c)

    def __abs__(self):
        """
        Applies the absolute value to the values of the pcf specified by self.

        Returns
        ------
        Pcf
            whose values are the absolute values of the given Pcf.
        """
        c = np.vstack((self.content[0], np.absolute(self.content[1])))
        return Pcf(c)

    def __truediv__(self, other):
        """
        Parameters
        ----------
        other: Real
            a non-zero real number.

        Returns
        -------
            Pcf, which is the outcome of the division.
        """
        if isinstance(other, numbers.Real):
            f = self.content
            c = np.vstack((f[0], f[1] / other))
            if other > 0:
                if isinstance(self, Pcnif):
                    return Pcnif(c)
                elif isinstance(self, Density):
                    return Density(c)
                else:
                    return Pcf(c)
            elif other < 0:
                return Pcf(c)
            else:
                raise ValueError("""we can not dived by 0""")
        else:
            raise ValueError("""A Pcf can only be divided by a non zero real. To divide by a PCF try 
                self*(other**(-1))""")

    def __rtruediv__(self, other):
        return other * (self**(-1))

    def integrate(self, start=-inf, end=inf):
        """
        Calculates the integral of the Pcf specified by self over the interval spanning between the start and the end.

        Parameters
        ----------
        start: real
            extended real number, optional, default -inf. It is the start of the integration.
        end: Real
            extended real number, optional, default inf. It is the end of the integration.

        Returns
        ------
        Real
            which is the result of the integration.
        """
        if start == end:
            return 0
        elif start < end:
            c = self.restrict(start, end)
        else:
            c = self.restrict(end, start)
        if c[1][-1] > 0:
            return inf
        elif c[1][-1] < 0:
            return -inf
        else:
            return sum(np.diff(c[0]) * c[1][:-1])

    def lp_distance(self, other, p=1, start=-inf, end=inf):
        """
        Calculates the lp_distance distance between the restrictions of the pcfs specified by the parameters
        self and other to the interval spanning between the start and the end.

        Parameters
        ----------
        other: Pcf
            a pcf to which the distance from the pcf specified by self is calculated.
        p: Real
            a real number, optional, default 1, assumed to be positive.
        start: Real
            extended real number, optional, default -inf. It specifies the start of the integration.
        end: Real
            extended real number, optional, default inf. It specifies the end of the integration.

        Returns
        ------
        Real
            which is the result of the integration.
        """
        if end == inf and self.content[1][-1] != other.content[1][-1]:
            return inf
        else:
            return (abs(self - other) ** p).integrate(start, end) ** (1 / p)      
        
    def approx(self, precision=-2, base=10):
        """
        Returns ndarray of dimension 2xn that approximates the pcf specified by self. The 0th row
        in the outcome consists of consecutive reals that differ by the step given by $base^{precision}$.
        The 1st row is given by the values of the pcf at the elements of the 0th row.

        Parameters
        ----------
        precision: int
            integer that specifies the power the base is raised to in order to define the step of the approximation.
        base: int
            integer whose power to the precision gives the step of the approximation.

        Returns
        -------
        ndarray.
        """
        precision = np.int_(precision)
        base = np.int_(base)
        if precision >= 0:
            step = base ** precision
        else:
            step = 1 / base**(-precision)    
        first = np.int_(np.floor(np.divide(self.content[0][0], step)))
        last = np.int_(np.ceil(np.divide(self.content[0][-1], step)))
        d = np.arange(first, last+1) * step
        ind = np.searchsorted(self.content[0], d, side="right")-1
        return np.vstack((d, self.content[1][ind]))
           

class Density(Pcf):
    r"""
    It is a subclass of the Pcf class and hence all the methods for the Pcf class can be used to densities.
    Furthermore the addition and multiplication of densities is a density.

    Objects in the Density class represent piecewise constant functions $f\colon [0,\infty) \to (0,\infty)$ defined
    on the non-negative real numbers with strictly positive values. To define a density, we need to specify an array
    with two rows. The 0-th row describes the domain and should start with 0 and consist of strictly increasing
    non-negative real numbers. The 1-st row describes the values of the function and should consist of strictly
    positive real numbers. These conditions are assumed, however not checked!

    Densities are important, since they are used to define contours which in turn are essential for specifying
    the length of bars. The method bar_length is the reason why we care about densities.

    Attributes
    ----------
    content: ndarray
        2xn ndarray of real numbers with the 0-th row, referred to as the domain, start with 0 and consists of
        strictly increasing non-negative real numbers, and the 1-st row consists of the values of the Pcf which
        are strictly positive.

    Raises
    ------
    ValueError
        The values for a Density have to be strictly positive with the domain starting at 0
    """
        
    def __init__(self, a):
        """
        Parameters
        ----------
        a: list[list[Real]], ndarray
            2xn array_like. It is assumed that $n>0$ and that the 0-th row, referred to as the domain, starts with
            0 and consists of strictly increasing non-negative real numbers. The second row consists of values and
            it is assumed that all the values are strictly positive.
        """
        super().__init__(a)
        if np.any(self.content[1] <= 0) or self.content[0][0] != 0:
            raise ValueError("""The values for a Density have to be strictly positive with the domain starting at 0""")

    def reverse_int(self, t):
        r"""
        For a density $f$ and non-negative real number $t$, the method produces a non-negative real number
        $x$ such that $\int_0^{x}f = t$.

        Parameters
        ----------
        t: Real

        Returns
        -------
        Real
        """
        cs = np.cumsum(np.diff(self.content[0]) * self.content[1][:-1])
        index = np.where(cs <= t)[0]
        if len(index) == 0:
            return t / self.content[1, 0]
        mi = max(index)+1
        e = self.content[1][mi]
        return self.content[0][mi] + (t - cs[mi-1]) / e

    def reverse_integrate(self, a, t):
        r"""
        For a density $f$ and non-negative real numbers $a$ and $t$, the method produces a non-negative real number
        $x\geq a$ such that $\int_a^{x}f = t$. It is the same as applying the following method:
        <code>f.reverse_int(t + self.integrate(0,a))</code>

        Parameters
        ----------
        a: Real
            real number specifying the start of the integration.
        t: Real
            real number specifying the result of the integration.

        Returns
        -------
        Real.
        """
        if t >= 0:
            if a >= self.content[0][-1]:
                return a + (t / (self.content[1][-1]))
            else:
                f = self.restrict(a, self.content[0][-1])
                cs = np.cumsum(np.diff(f[0]) * f[1][:-1])
                index = np.where(cs <= t)[0]
                if len(index) == 0:
                    return a + (t / f[1][0])
                else:            
                    mi = max(index)
                    e = f[0][mi + 1]
                    return f[0][mi + 1] + (t - cs[mi]) / self.evaluate(e)
        raise ValueError("""Reverse Integrate can only be done if the second parameter t is non negative""")

    def dist_contour(self, a, t, truncation=inf):
        r"""
        Contours describe two type of actions of the additive monoid of non-negative real numbers $[0,\infty)$ on
        the poset of extended non-negative real numbers $[0,\infty]$. One action is of so called distance kind.
        This method produces the outcome of the action of t on a of the distance kind contour induced by the density.
        If $f$ denotes the density function, then this action is computed as follows:
        -   find $x$ so that $\int_a^{x}f = t$, which is exactly the value obtained  by the
            method <code>f.reverse_integrate(a, t)</code>,
        -   return x if x < truncation,
        -   return inf if x>= truncation.

        Parameters
        ----------
        a: Real
            extended real number specifying an element in the poset $[0,\infty]$. It has to be non-negative, which
            however is not checked.
        t: Real
            real number specifying an element in the additive monoid $[0,\infty)$. It has to be non-negative, which
            however is not checked.
        truncation: Real
            extended real number specifying the truncation. It has to be positive, which however is not checked.

        Returns
        -------
        Real
            extended real number which is the outcome of the action of t on a with respect to the contour of
            the distance kind induced by the density and truncated at truncation.
        """
        ri = self.reverse_integrate(a, t)
        if ri < truncation:
            return ri
        return inf

    def area_contour(self, a, t, truncation=inf):
        r"""
        Contours describe two type of actions of the additive monoid of non-negative real numbers $[0,\infty)$ on
        the poset of extended non-negative real numbers $[0,\infty]$. One action is of so called distance kind.
        This method produces the outcome of the action of t on a of the area kind contour induced by the density.
        If $f$ denotes the density function, then this action is computed as follows:
        -   find $y$ so that $\int_0^{y}f = a$, which is exactly the value obtained  by the
            method <code>f.reverse_int(a)</code>,
        -   let  $x = \int_0^{y+t}f$,
        -   return x if x < truncation,
        -   return inf if x>= truncation.

        Parameters
        ----------
        a: Real
            extended real number specifying an element in the poset $[0,\infty]$. It has to be non-negative, which
            however is not checked.
        t: Real
            real number specifying an element in the additive monoid $[0,\infty)$. It has to be non-negative, which
            however is not checked.
        truncation: Real
            extended  real number specifying the truncation. It has to be positive, which however is not checked.
        Returns
        -------
        Real
            extended real number which is the outcome of the action of t on a with respect to the contour of
            the area kind induced by the density and truncated at truncation.
        """
        y = self.reverse_int(a)
        c_at = a + self.integrate(y, y + t)
        if c_at < truncation:
            return c_at
        return inf

    def bar_length(self, bar, contour_kind="dist", truncation=inf):
        r"""
        It calculates the length of a bar with respect to the specified contour_kind and truncation.
        Let $f$ be the chosen density.
        -   for the contour_kind "dist", the length of a bar $[s,e)$ is given by the integral
            $\int_s^{ \text{min}(e,\text{truncation})} f$.
        -   for the contour_kind "area", the length of a bar $[s,e)$ is given by the difference $x_e-x_s$, where
            -   $x_s$ is the solution of the equation $\text{min}(s, \text{truncation})=\int_0^{x_s} f$
            -   $x_e$ is the solution  of the equation $\text{min}(e, \text{truncation}) =\int_0^{x_e} f$.

        Parameters
        ----------
        bar: list[Real], ndarray
            1x2 array like of real numbers describing a bar.
        contour_kind: string
            optional, default "dist". It specifies which contour type to use. Among possible choices are
            "dist" and "area".
        truncation: Real
            optional, default inf. Its a non-negative extended real number specifying the truncation of the contour.

        Returns
        ------
        Real.
        """
        if len(bar) == 0:
            return np.nan
        if np.isnan(bar[1]) or bar[1] == inf:
            return inf
        end = np.min((bar[1], truncation))
        if bar[0] >= end:
            return 0
        if contour_kind == "dist":
            return self.integrate(bar[0], end)
        if contour_kind == "area":
            return self.reverse_int(end) - self.reverse_int(bar[0])
        raise ValueError("""you need to choose a contour_type to be either "dist" or "area" """)


class Pcnif(Pcf):
    """
    Pcnif stands for piece wise constant non increasing functions. It is a subclass of the Pcf class and hence all
    the methods for the Pcf class can be used to pcnifs. Furthermore the addition, multiplication of a pcnif is a pcnif.

    Objects in this class represent piece wise constant non increasing functions defined on the non-negative real
    numbers with values in non-negative real numbers. To define a Pcnif object, we need to specify an array with
    two rows. The 0-th row describes the domain and should start with 0 and consists of strictly increasing non-negative
    real numbers. The 1-st row describes the values of the function and should consists of non-increasing non-negative
    real numbers. These conditions are assumed, however not checked!

    Attributes
    ----------
    content: ndarray
        2xn ndarray whose values are real numbers. Its 0-th row, referred to as the domain, starts with 0 and
        consists of strictly increasing non-negative reals. Its 1-st row consists of the values of the pcnif which
        are are assumed to be non increasing."""
    
    def __init__(self, a):

        """
        Parameters
        ----------
        a: list[list[float]], ndarray
            it is a 2xn array like of real numbers with n>0.  It is assumed that the 0-th row start with 0 and consists
            of strictly increasing non-negative real numbers, and 1-st row consists of non-increasing non-negative
            real numbers.
        """
        super().__init__(a)
 
    def _interl(self, other):
        if self.content[1][-1] < other.content[1][-1]:
            return inf
        fd = self.content[0]
        fv = self.content[1]
        gd = other.content[0]
        gv = other.content[1]
        intervals = np.array([])
        i = 0
        while i < len(gd) - 1:
            ind = np.searchsorted(-fv, -gv[i], side='right')
            intervals = np.append(intervals, gd[i + 1] - fd[ind])
            i += 1
        return np.amax(intervals)
            
    def interleaving_distance(self, other):
        """
        Calculates the interleaving_distance between the pcnifs specified by self and other.

        Parameters
        ----------
        other: Pcnif.

        Returns
        ------
        Real.
        """
        return max(self._interl(other), other._interl(self))
            
    def flatness(self):
        """
        Let $f$ be a Pcnif object and $c$ be its content. Then for avery value $v$ of $f$ bigger or equal to $c[1,-1]+1$
        we consider the preimages $f^{-1}([v-1,v))$. These preimages are intervals. This method gives the beginning of
        the longest among these intervals. For example if $f$ has a single value, then $0$ is returned.

        Returns
        -------
        tuple
            consisting of a real number and 1D ndarray. The real number is the beginning of the longest interval of the
            form $f^{-1}([v-1,v))$  where $v$ is any value of $f$ bigger or equal to <code>f.content[1,-1]+1</code>.
            The 1D np ndarray consists of the length of the intervals of the form $f^{-1}([v-1,v))$.
        """
        f = self.simplify()
        v = f[1, :]
        d = f[0, :]
        if len(v) == 1:
            return 0, np.array([])
        else:
            i = 0
            life = np.array([])
            while v[i] >= v[-1]+1:
                dif = np.where(v > v[i]-1)[0]
                if len(dif) > 0:
                    life = np.append(life, d[dif[-1]+1] - d[i])
                else:
                    life = np.append(life, 0)
                i = i+1
            der = np.argmax(life)
            return d[der], life

    def diff(self):
        d = self.simplify()[0, :]
        if len(d) == 1:
            return 0
        else:
            life = np.diff(d)
            der = np.argmax(life)
            return d[der]


class Distribution(object):
    r"""
    Objects in this class represent functions $f\colon (-\infty,\infty)\to [0,\infty)$.
    To define a Distribution we need to specify a list of dictionaries whose keys are "name", "arg" and "coef".
    The value  at the key "name" is a string and can be either "uniform", "normal",
    "linear", or "plf". The value at the key "arg" consists of corresponding arguments. The values at the key "coef"
    are strictly positive reals. We think about a Distribution as a linear combination of four type of Distribution
    "uniform", "normal", "linear", and "plf". The "coef"  specifies the coefficients of the linear combination.

    Attributes
    ----------
    content: tuple[dict].
    """

    def __init__(self, content):
        r"""
        Parameters
        ----------
        content: str, list, ndarray, dict, tuple
            it can be either:
            -   string which is either "uniform" or "normal" or "linear". In this case the input is converted into
                the following tuple containing only one dictionary
                <code> ({"name":"uniform", "arg":[0,1], "coef":1},) </code>
            -   list of the form $[\text{name},A]$ where name is a string as in the previous point and $m$ and $d$
                are real numbers. If the name is "uniform", then A is an array of the form [b,e] describing the
                support of the uniform distribution. If the name is "normal" or "linear", then A is an array of the
                form [m,d] where we refer to m as the center and d as the deviation. If the name is "plf", then
                A is an 2xn array like of real numbers where the 0th row is strictly increasing and the 1st row consists
                of non negative real numbers. Each column describes what value (position 1) is assign to an element in
                the domain (position 0). Then these assignments are extended to the entire line in a piece wise linear
                way. In this case the input is converted into one element tuple consisting of the following dictionary:
                <code> ({"name":name, "arg": A, "coef":1},) </code>".
            -   list of the form $[\text{name},A,c]$. In this case the input is converted into one element tuple
                consisting of the following dictionary: <code> ({"name":name, "arg": A, "coef":c},) </code>".
            -   dictionary with 1 to 3 keys:  "name", and possibly "arg", and possibly "coef" (the key "name" should
                be always included, and if there are only 2 keys, then the second one has to be "arg").
            -   list of dictionaries, each of them with 1 to 3 keys:  "name", and possibly "arg", and possibly "coef"
                as before.
        """
        if isinstance(content, str):
            self.content = tuple([_str_to_distribution(content)])
        elif isinstance(content, dict):
            self.content = tuple([_dict_to_distribution(content)])
        elif isinstance(content, (np.ndarray, list, tuple)) and isinstance(content[0], str):
            self.content = tuple([_list_string_to_distribution(content)])
        elif isinstance(content, (np.ndarray, list, tuple)) and isinstance(content[0], dict):
            self.content = _list_dict_to_distribution(content)
        else:
            raise ValueError("""Wrong parameters """)

    def evaluate(self, x):
        """
        The method valuates the Distribution at given values so that the total sum is either 0 or 1.
        That is why we think about the outcome as probabilities.

        Parameters
        ----------
        x: Real, list[Real], ndarray
            either a real number or an array like of real numbers. It specifies the values at which the Distribution
            is evaluated on.

        Returns
        ------
        boolean False
            in the case the values of the Distribution at each entry in x is 0, or
        real number 1
            in the case x is a real number for which the value of the Distribution is positive, or
        1D ndarray of real numbers that sum up to 1
            in the case x is a 1D array like containing a value for which the evaluation of the Distribution
            is positive.
        """
        if isinstance(x, numbers.Real):
            xx = np.array([x], dtype='float64')
        else:
            xx = np.asarray(x, dtype='float64')
        m = 0
        for d in self.content:
            v = _evaluate_single(d["name"], d["arg"], xx) * d["coef"]
            m += v
        if np.sum(m) == 0:
            return False
        else:
            return np.asarray(m / np.sum(m), dtype='float64')

    def plot(self, x_min, x_max, precision=-2, base=10, ax=None, **kwargs):
        r"""
        Plots the Distribution.

        Parameters
        ----------
        x_min: Real
            It specifies the start point for the plot's domain.
        x_max: Real
            It specifies the end point for the plot's domain.
        precision: int
            optional, default -2. It describes the power the base is raised to.
        base: int
            optional, default 10. Specifies the base whose power describes the steps for the discretization of the
            distribution. The Distribution is evaluated at points starting at x_min and moving along steps of size
            $\text{base}^{\text{precision}}$. Thus default step is $1/100$.
        ax: None, pyplot axes
            optional, default None. In case it is None  plt.plot is produced. Otherwise a plt.axes is
            produced.
        **kwargs: additional parameters
            the keyword arguments for the returned axes class, which include color, linewidth.
        Returns
        ------
        pyplot plot or axes of the Distribution.
        """
        if precision >= 0:
            step = base ** precision
        else:
            step = 1 / base ** (-precision)
        x = np.arange(x_min, x_max, step)
        y = self.evaluate(x)
        if isinstance(y, bool) and y is False:
            y = np.zeros(len(x))
        else:
            y = y / (np.sum(y) * step)
        if ax is None:
            ax = plt
        return ax.plot(x, y, **kwargs)

    def __add__(self, other):
        """
        Adds two distributions to form a new distribution. We can use  <code>f + g</code> or <code>f.add(g)</code>.

        Parameters
        ----------
        other: Distribution
            the distribution added to self.

        Returns
        ------
        Distribution
            which is the result of the addition.
        """
        return Distribution(self.content + other.content)

    def __radd__(self, other):
        return self + other

    def __mul__(self, r):
        """
        Multiplies a distribution by a positive real number to form a new distribution. We can use
        <code>f * r</code> or <code>r * f</code> or <code>f.mul(r)</code>.

        Parameters
        ----------
        r: Real
            which has to be positive.

        Returns
        ------
        Distribution
            which is the result of the multiplication.
        """
        out = []
        for d in self.content:
            dd = {"name": d["name"], "arg": d["arg"], "coef": d["coef"] * r}
            out.append(dd)
        return Distribution(out)

    def __rmul__(self, r):
        return self * r

    def coef_change(self, c):
        """
        Changes the coefficients of the distribution.

        Parameters
        ----------
        c: list[Real], ndarray
            is an array like consisting of non-negative real numbers with at least one of them non-zero.
            The length of c is assumed to be equal to the length of self.content
            The indexes corresponding to coordinates in c with the zero value are removed.

        Returns
        ------
        Distribution
            with new coefficients.
        """
        out = []
        i = 0
        while i < len(c):
            d = self.content[i]
            coef = c[i]
            if coef != 0:
                dd = {"name": d["name"], "arg": d["arg"], "coef": coef}
                out.append(dd)
            i += 1
        return Distribution(out)


##########################################
##########################################
##########################################


class Contour(object):
    
    def __init__(self, d, kind, truncation):
        self.density = d
        self.kind = kind
        self.truncation = truncation


# Producing standard distributions

def moving_distribution(diameter, size, n, name="uniform"):

    """
    Produces a dictionary of n uniform distributions over intervals of length given by the size and equally spaced over
    the interval from 0 to diameter.

    Parameters
    ----------
    diameter:  Real
        a real number assumed to be positive. All the distributions will be spanning across the
        interval $[0,\text{diameter}]$.
    size: Real
        a real number assumed to be positive.
    n:  int
        an integer assumed to be positive and describes number of distributions obtained.
    name:  str
        optional, default "uniform". Can be either "uniform", "normal", or "linear". Specifies the kind of distribution.


    Returns
    ------
    dictionary
        of distributions, indexed by keys of the form "d_m_i" where i ranges from 0 to n-1.
        To each such key a Distribution with the parameter [name,[m,size/2]], over an interval of
        length size is assign."""

    out = {}
    b = np.linspace(0, diameter - size, num=n)
    i = 0
    while i < n:
        m = b[i] + (size/2)
        d = size/2
        out["d_m_"+str(i)] = Distribution([name, [m, d]])
        i += 1
    return out


def increasing_distribution(diameter, n, name="uniform"):
    """
    Produces a dictionary of n distributions over intervals of increasing lengths starting from 0 and whose
    endings increasingly grow to the parameter diameter.

    Parameters
    ----------
    diameter:  Real
        a real number assumed to be positive. All the distributions will be spanning across the
        interval $[0,\text{diameter}]$.
    n:  int
        an integer assumed to be positive and describes number of distributions obtained.
    name:  str
        optional, default "uniform". Can be either "uniform", "normal", or "linear". Specifies the kind of distribution.

    Returns
    ------
    dictionary
        of distributions, indexed by keys of the form "d_inc_i" where i ranges from 0 to N-1.
        To each such key a Distribution is assigned.

        Indexed by keys of the form "d_inc_i" where i ranges from 0 to n-1. To each such key a uniform Distribution over
        the interval $[0, (i+1)diameter/N]$ is assign."""

    out = {}
    b = np.linspace(0, diameter, num=n + 1)
    i = 1
    while i < n+1:
        m = b[i]/2
        d = b[i]/2
        out["d_inc_" + str(i-1)] = Distribution([name, [m, d]])
        i += 1
    return out


def decreasing_distribution(diameter, n, name="uniform", e=None):
    """
    Produces a dictionary of n  distributions over intervals of increasing lengths with the starting point
    increasing and ending given by the parameter diameter.

    Parameters
    ----------
    diameter:  Real
        a real number assumed to be positive. All the distributions will be spanning across the
        interval $[0,\text{diameter}]$.
    n:  int
        an integer assumed to be positive and describes number of distributions obtained.
    name:  str
        optional, default "uniform". Can be either "uniform", "normal", or "linear". Specifies the kind of distribution.
    e:  None or not None
        optional, default None. It indicates if we consider the enter interval.

    Returns
    ------
    dictionary
        of distributions, indexed by keys of the form "d_dec_i" where i ranges from 0 to n-1.
        To each such key a Distribution is assigned."""

    out = {}
    b = np.linspace(0, diameter, num=n + 1)
    if e is None:
        i = 0
    else:
        i = 1
    while i < n:
        m = (diameter + b[i]) / 2
        d = (diameter - b[i]) / 2
        out["d_dec_" + str(i)] = Distribution([name, [m, d]])
        i += 1
    return out


###########################################################
###########################################################


def _evaluate_single(name, arg, x):
    """x is assumed to be an ndarray"""
    if name == "uniform":
        dis = np.greater_equal(x, arg[0]) * np.less_equal(x, arg[1]) * 1.0
        return dis
    elif name == "normal":
        ex = (-1 / 2) * ((x - arg[0]) / arg[1])**2
        c = 1 / (arg[1] * (2 * np.pi)**0.5)
        dis = c * np.exp(ex)
        return dis
    elif name == "linear":
        d1 = np.fmax(arg[1] - np.absolute(x - arg[0]), 0)
        d2 = np.amax(x) / (2 * arg[1])
        dis = d1 * d2
        return dis
    elif name == "plf":
        ind = np.searchsorted(arg[0], x)
        c1 = np.diff(arg[1]) / np.diff(arg[0])
        c = np.concatenate(([0], c1, [0]))[ind]
        b = np.concatenate(([0], arg[0]))[ind]
        v = np.concatenate(([arg[1][0]], arg[1]))[ind]
        dis = c * (x - b) + v
        return dis
    else:
        raise ValueError("""You can choose only between uniform, normal, linear, or plf distributions""")


def _str_to_distribution(name):
    """name is assumed to be a string "uniform", "normal", or "linear" """
    c = {"name": name, "coef": 1.0}
    if name == "uniform":
        c["arg"] = [-inf, inf]
    elif name in ["normal", "linear"]:
        c["arg"] = [0.0, 1.0]
    else:
        raise ValueError("""Wrong description of a Distribution""")
    return c


def _dict_to_distribution(d):
    """d is assumed to be a dictionary with keys "name", "arg", and possibly "coef" """
    if len(d.keys()) == 1:
        return _str_to_distribution(d["name"])
    c = d
    if len(d.keys()) == 2:
        c["coef"] = 1.0
    return c


def _list_string_to_distribution(a):
    """a is assumed to be a list of length 2 or 3, starting with a string "uniform", or "normal" or "linear
    or "plf"."""
    c = {"name": a[0], "arg": a[1]}
    if len(a) == 2:
        c["coef"] = 1
        return c
    c["coef"] = a[2]
    return c


def _list_dict_to_distribution(a):
    """a is assumed to be a list of dictionaries"""
    out = []
    for d in a:
        out.append(_dict_to_distribution(d))
    return tuple(out)
