# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 18:25:07 2018

@author: Steven Xu
"""
#%%
import numpy as np
#%% simulation variables
num_points = 4
#ring_radius = 1.5e11*1e-5
##ring_radius = 1
#G = 6.67408e-11
#M = 1.989e30
#m = 5.972e24/num_points
#young_mod = 1e2
#X_section_area = 1.3e14
ring_radius = 1
#G = 6.67408e-11
G = 1e-11
M = 1e9
m = 1
young_mod = 1e0
X_section_area = 100
k = 0
L0=1
#%%
def shift_array_cc(XYZ): #this pair of functions is to easily get n+1 and n-1 indices
    return np.append(XYZ[1:,::],[XYZ[0,::]],0)

def shift_array_c(XYZ):
    return np.append([XYZ[-1,::]], XYZ[0:-1,::],0 )

#%% generating a circle of points
XYZ = np.zeros([num_points,3])
V = np.zeros([num_points,3])

for i in range(num_points):
    theta = 2 * np.pi  * i/num_points
    XYZ[i] = [ring_radius*np.cos(theta), ring_radius*np.sin(theta),0]


XYZ0 = np.copy(XYZ)
XYZ_shifted = shift_array_cc(XYZ)
L0 = np.linalg.norm(XYZ-XYZ_shifted, axis=1, keepdims = True)
#%%
XYZm = shift_array_cc(XYZ)
XYZp = shift_array_c(XYZ)
X = XYZ[:,0:1]; Y = XYZ[:,1:2]; Z=XYZ[:,2:]
Xm = XYZm[:,0:1]; Ym = XYZm[:,1:2]; Zm=XYZm[:,2:]
Xp = XYZp[:,0:1]; Yp = XYZp[:,1:2]; Zp=XYZp[:,2:]
R = np.linalg.norm(XYZ, axis=1, keepdims = True)
omega2 = k/m
Xs = np.repeat(X, num_points, axis=1)
Ys = np.repeat(Y, num_points, axis=1)
Zs = np.repeat(Z, num_points, axis=1)
XTs = np.transpose(Xs)
YTs = np.transpose(Ys)
ZTs = np.transpose(Zs)
Rn = ((Xs-XTs)**2 + (Ys-YTs)**2 + (Zs-ZTs)**2)**.5
XX = np.nan_to_num((Xs - XTs)/Rn)
YY = np.nan_to_num((Ys - YTs)/Rn)
ZZ = np.nan_to_num((Zs - ZTs)/Rn)
Lm = np.linalg.norm(XYZ-XYZm, axis=1, keepdims = True)
Lp = np.linalg.norm(XYZ-XYZp, axis=1, keepdims = True)
UgX = np.sum(XX,axis=1, keepdims = True)
UgY = np.sum(YY, axis=1, keepdims = True)
UgZ = np.sum(ZZ, axis=1, keepdims = True)
DX2 = -G*(M*X/R + m * UgX) + omega2*(2*X - (Xm + Xp) - L0*((X-Xm)/Lm + (X-Xp)/Lp))
DY2 = -G*(M*Y/R + m * UgY) + omega2*(2*Y - (Ym + Zp) - L0*((Y-Ym)/Lm + (Y-Yp)/Lp))
DZ2 = -G*(M*Z/R + m * UgZ) + omega2*(2*Z - (Zm + Yp) - L0*((Z-Zm)/Lm + (Z-Zp)/Lp))
A = np.concatenate((DX2, DY2, DZ2), axis = 1)