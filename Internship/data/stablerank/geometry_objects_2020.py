#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr  4 15:32:33 2019

@author: Wojciech chacholski, 2019

Copyright Wojciech chacholski, 2019
This software is to be used only for activities related  with TDA group at KTH 
and TDA course SF2956 at KTH
"""
import numpy as np
inf=float("inf")



import stablerank.srank as sr

import scipy.stats as st

import itertools as it




############
# CIRCLE
############    
class circle(sr.EucObject):
    
    def __init__(self,arg, size, error = 0):
        c = arg[0] # center
        r = arg[1] # radious
        T = np.random.uniform(high = 2*np.pi, size = size)
        Y = np.sin(T) * (r) + c[1]
        X = np.cos(T) * (r) + c[0]
        sd = error * 0.635
        pdf = st.norm(loc = [0,0],scale =(sd,sd))
        N = pdf.rvs((size, 2))
        self.points = N+np.vstack((X, Y)).transpose()
        self.center = c
        self.radious = r
        self.arg = arg
        self.size = size
        self.error = error
        self.kind = "circle"
        self.dim = 2
        self.area=np.pi * r * r
        self.length=2 * np.pi * r

############
# lp CIRCLE
############    
class lp_circle(sr.EucObject):
    
    def __init__(self,arg, size, error=0):
        c=arg[0] # center
        r=arg[1] # radious
        p=arg[2]
        T = np.random.uniform(high=2*np.pi, size = size)
        
        Y= np.sin(T)*(r)* 1/(  ( (np.absolute(np.sin(T)))**p+(np.absolute(np.cos(T)))**p)**(1/p) )+c[1]
        X= np.cos(T)*(r)* 1/(  ( (np.absolute(np.sin(T)))**p+(np.absolute(np.cos(T)))**p)**(1/p) )+c[0]
        sd=error*0.635
        pdf=st.norm(loc = [0,0],scale=(sd,sd))
        N=pdf.rvs((size, 2))
        self.points = N+np.vstack((X,Y)).transpose()
        self.center=c
        self.radious=r
        self.arg=arg
        self.size = size
        self.error=error
        self.kind="lp_circle"
        self.dim=2
        
#############
# DISC
#############
class disc(sr.EucObject):
    def __init__(self, arg, size):
        c = arg[0] # center
        r = arg[1] # radious
        u = np.random.normal(0, 1, size = size)
        v = np.random.normal(0, 1, size = size)
        norm = (u * u + v * v)**0.5
        rad = r*np.random.rand(size)**0.5
        X=(rad * u / norm) + c[0]
        Y=(rad * v / norm) + c[1]            
        self.points = np.vstack((X, Y)).transpose()
        self.center = c
        self.radious = r
        self.arg = arg
        self.size = size
        self.kind = "disc"
        self.dim = 2
        self.area = np.pi * r * r
        self.length = 2 * np.pi * r

######################
# SQAURE_BOUNDARY
######################
class square_boundary(sr.EucObject):
    
    def __init__(self, arg, size, error = 0):
        c = arg[0] # center
        r = arg[1] # radious
        T = np.random.uniform(high=8 * r, size=size)
        X = np.array([])
        Y = np.array([])
        for t in T:
            if t >= 0 and t < 2 * r:
                X = np.append(X, c[0] - r + t)
                Y = np.append(Y, c[1] + r)
            elif t >= 2 * r and t < 4 * r:
                X = np.append(X, c[0] + r)
                Y = np.append(Y, c[1] + r - t + 2*r)
            elif t >= 4 * r and t < 6 * r:
                X = np.append(X, c[0] + r - t + 4*r)
                Y = np.append(Y, c[1] - r)
            else:
                X = np.append(X, c[0] - r)
                Y = np.append(Y, c[1]- r + t - 6*r)
        sd = error * 0.635
        pdf = st.norm(loc = [0, 0], scale = (sd, sd))
        N=pdf.rvs((size, 2))
        self.points = N + np.vstack((X, Y)).transpose()
        self.center = c
        self.radious = r
        self.arg = arg
        self.size = size
        self.error = error
        self.kind = "square_boundary"
        self.dim = 2
        self.area = 4 * r * r
        self.length = 8 * r                

#############
## Normal_point
############# 
class normal_point(sr.EucObject):
    def __init__(self, arg, size, error = [[1,0],[0,1]]):
        c=arg # point
        points=np.random.multivariate_normal(c, error, size=size)
        self.points = points
        self.center = c
        self.arg = arg
        self.size = size
        self.error = error
        self.kind = "normal_point"
        self.dim = 2
        self.length = 0




############
# close_PATH
############
class closed_path(sr.EucObject):

    def __init__(self, arg, size, error):   
        vertices = np.asarray(arg)
        number_vertices = len(vertices)
        L1 = np.linalg.norm(vertices[1:, :] - vertices[:-1, :], axis = 1)
        L=np.concatenate([L1, np.array([np.linalg.norm(vertices[0] - vertices[-1])])])
        accum_L = np.asarray(list(it.accumulate(L)))
        T = np.random.uniform(high=accum_L[-1], size = size)        
        points=np.empty([0, 2])
        for t in T:
            index=np.searchsorted(accum_L, t)
            coef = (accum_L[index] - t) / (L[index])
            if index == number_vertices - 1:
                points = np.vstack((points, (coef * vertices[0] + (1 - coef) * vertices[-1])))
            else:
                points = np.vstack((points, (coef * vertices[index + 1] + (1 - coef) * vertices[index])))
        N=np.empty([0, 2])
        sd = error * 0.635
        pdf = st.norm(loc = [0, 0], scale = (sd, sd))
        N=pdf.rvs((size, 2))
        self.points = N+points
        self.arg = arg
        self.size = size
        self.error = error
        self.kind = "closed_path"
        self.dim = 2
        self.length = sum(L1) + np.linalg.norm(vertices[0] - vertices[-1])

############
# open_PATH
############
class open_path(sr.EucObject):

    def __init__(self, arg, size, error):   
        vertices = np.asarray(arg)
        L = np.linalg.norm(vertices[1:, :]-vertices[:-1, :], axis = 1)
        accum_L = np.asarray(list(it.accumulate(L)))
        T = np.random.uniform(high=accum_L[-1],size = size)        
        points = np.empty([0, 2])
        for t in T:
            index=np.searchsorted(accum_L,t)
            coef=(accum_L[index]-t)/(L[index])
            points=np.vstack((points,(coef*vertices[index+1] + (1-coef) * vertices[index])))
        N=np.empty([0, 2])
        sd=error*0.635
        pdf=st.norm(loc=[0,0],scale=(sd,sd))
        N=pdf.rvs((size,2))
        self.points=N+points
        self.arg=arg
        self.size = size
        self.error=error
        self.kind="open_path"
        self.dim=2
        self.length=sum(L)     


#############
## Uniform Noise
############# 
class uniform_noise(sr.EucObject):
    def __init__(self, arg, size):
        """arg is a list (x_max, x_min, y_max, y_min) """        
        X = (np.random.random(size) * (arg[0]-arg[1])) + arg[1]
        Y = (np.random.random(size) * (arg[3]-arg[2])) + arg[2] 
        self.arg = arg
        self.kind = "uniform_noise"
        self.dim=2
        self.points = np.vstack((X,Y)).transpose()
        self.size = size

########################################
#### FUNCTIONS        
########################################

def shape(obj_arg=("circle",([0,0],1)), size = 100, error=0.1):
    object_kind=obj_arg[0]
    arg=obj_arg[1]
    if object_kind=="circle":
        return circle(arg, size, error)
    elif object_kind=="lp_circle":
        return lp_circle(arg, size, error)
    elif object_kind=="disc":
        return disc(arg, size)
    elif object_kind=="square_boundary":
        return square_boundary(arg, size, error)
    elif object_kind=="normal_point":
        return  normal_point(arg, size, error)
    elif object_kind=="open_path":
        return  open_path(arg, size, error)
    elif object_kind=="closed_path":
        return  closed_path(arg, size, error)
    elif object_kind=="uniform_noise":
        return uniform_noise(arg, size)
    else:
        raise ValueError("There is no plane shape with that name") 


if __name__ == "__main__":
    object_kind="uniform_noise"
    arg=(5,-1,30,1)
    size = 300
    C=shape((object_kind,arg))
    C.plot()
    
