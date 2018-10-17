# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 21:20:03 2018

@author: Steven Xu
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%% simulation variables
num_points = 20
#ring_radius = 1.5e11*1e-5
##ring_radius = 1
#G = 6.67408e-11
#M = 1.989e30
#m = 5.972e24/num_points
#young_mod = 1e2
#X_section_area = 1.3e14
ring_radius = 1
#G = 6.67408e-11
G = 1e-11*0
#G=0
M = 1e9
m = 1
young_mod = 1e0
X_section_area = 100
visual_range = ring_radius*1.2e0
k = 0
L0=1
tmax = 5
t=0
t_step = 1e0
#%% functions
#def motion(IJK):
    
def shift_array_cc(XYZ): #this pair of functions is to easily get n+1 and n-1 indices
    return np.append(XYZ[1:,::],[XYZ[0,::]],0)

def shift_array_c(XYZ):
    return np.append([XYZ[-1,::]], XYZ[0:-1,::],0 )

def lagrangian(XYZ):
    global L0
    XYZm = shift_array_cc(XYZ)
    XYZp = shift_array_c(XYZ)
    X = XYZ[:,0:1]; Y = XYZ[:,1:2]; Z=XYZ[:,2:]
    Xm = XYZm[:,0:1]; Ym = XYZm[:,1:2]; Zm=XYZm[:,2:]
    Xp = XYZp[:,0:1]; Yp = XYZp[:,1:2]; Zp=XYZp[:,2:]
    R = np.linalg.norm(POS, axis=1, keepdims = True)
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
    DX2 = -G*(M*X/R + m * UgX) + -omega2*(2*X - (Xm + Xp) - L0*((X-Xm)/Lm + (X-Xp)/Lp))
    DY2 = -G*(M*Y/R + m * UgY) + -omega2*(2*Y - (Ym + Yp) - L0*((Y-Ym)/Lm + (Y-Yp)/Lp))
    DZ2 = -G*(M*Z/R + m * UgZ) + -omega2*(2*Z - (Zm + Zp) - L0*((Z-Zm)/Lm + (Z-Zp)/Lp))
    A = np.concatenate((DX2, DY2, DZ2), axis = 1)
    return A
#%% generating a circle of points
POS = np.zeros([num_points,3])
V = np.zeros([num_points,3])
v = (G*M*m/ring_radius)**.5
#%%
for i in range(num_points):
    theta = 2 * np.pi  * i/num_points
    POS[i] = [ring_radius*np.cos(theta), ring_radius*np.sin(theta),0]
    V[i] = [v*-np.sin(theta), v*np.cos(theta),0] #orbital
#    V[i] = [np.cos(theta), np.sin(theta),0] #radial, for testing
V[abs(V)<1e-15]=0
POS[abs(POS)<1e-15]=0
#V[0] = [1,0,0]
#V[1] = [-1,0,0]
#V=0
V0 = np.copy(V)
POS0 = np.copy(POS)
POS_shifted = shift_array_cc(POS)
#links go in same direction as points -- counterclockwise
L0 = np.linalg.norm(POS-POS_shifted, axis=1, keepdims = True)
#k = young_mod*X_section_area/L0[0]

#%% Plot Parameters
fig = plt.figure(figsize=(10,8))
ax = p3.Axes3D(fig)
ax.set_xlim3d([-visual_range, visual_range])
ax.set_ylim3d([-visual_range, visual_range])
ax.set_zlim3d([-visual_range, visual_range])
ax.set_facecolor('black')
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')
ax.tick_params(colors='white')
#%%
#print([dat[:,:] for dat in [POS,POS_shifted]])

#%% Create the List of Points
lines = [ax.plot([dat[0]],[dat[1]],[dat[2]]) for dat in POS]
sun = [ax.plot([0],[0],[0])]
points = [ax.plot([dat[0]], [dat[1]], [dat[2]]) for dat in POS]

plt.setp(sun, marker='.', color='orange',markersize=50)
#V = np.zeros((num_points, 3))
color_increment = 1/num_points
colors = [0,0,.49]
for index, point in enumerate(points):
    plt.setp(point, marker='.', markersize=20, color = colors)
    points[index] = point[0]
    colors = [colors[0]+color_increment, colors[1]+color_increment, colors[2]+color_increment/2]
colors = [0,0,.49]
for index, line in enumerate(lines):
    plt.setp(line, color=colors, linewidth=10)
    lines[index] = line[0]
    colors = [colors[0]+color_increment, colors[1]+color_increment, colors[2]+color_increment/2]


#%% animation loop (matplotlit)
#def animate(t):
#    global V; global POS
#    POS = POS + V*t_step
#    V = V + (lagrangian(POS)*t_step)
#    POScc = shift_array_cc(POS)
#    node_distances = POS-POScc
#    for index, point in enumerate(points):
#         point.set_xdata(POS[index,0])
#         point.set_ydata(POS[index,1])
#         point.set_3d_properties(POS[index,2])
#    for index, line in enumerate(lines):
##        if 3* L0[index,0] > np.linalg.norm(node_distances[index]):
#            line.set_xdata([POS[index,0], POScc[index,0]])
#            line.set_ydata([POS[index,1], POScc[index,1]])
#            line.set_3d_properties([POS[index,2], POScc[index,2]])
##        else:
##            line.set_xdata([0],[0])
##            line.set_ydata([0],[0])
##            line.set_3d_properties([0],[0])
##            line.set_alpha(0)
#    return points, lines
#%% data generation loop

while t < tmax:
    POS = POS + V*t_step
    V = V + (lagrangian(POS)*t_step)

#%% animation loop (MoviePy)
def animate(t):
    global V; global POS
    POS = POS + V*t_step
    V = V + (lagrangian(POS)*t_step)
    POScc = shift_array_cc(POS)
    node_distances = POS-POScc
    for index, point in enumerate(points):
         point.set_xdata(POS[index,0])
         point.set_ydata(POS[index,1])
         point.set_3d_properties(POS[index,2])
    for index, line in enumerate(lines):
#        if 3* L0[index,0] > np.linalg.norm(node_distances[index]):
            line.set_xdata([POS[index,0], POScc[index,0]])
            line.set_ydata([POS[index,1], POScc[index,1]])
            line.set_3d_properties([POS[index,2], POScc[index,2]])
#        else:
#            line.set_xdata([0],[0])
#            line.set_ydata([0],[0])
#            line.set_3d_properties([0],[0])
#            line.set_alpha(0)
    return mplfig_to_npimage(fig)

#%% Test block
#for index, point in enumerate(points):  
##    point.set_data(POS[index,0:2])
#    point.set_3d_properties()

#for index, line in enumerate(lines):
#    line.set_xdata([POS[index,0], POS_shifted[index,0]])
#    line.set_ydata([POS[index,1], POS_shifted[index,1]])
#    line.set_3d_properties([POS[index,2], POS_shifted[index,2]])

#%% Animation and saving block
ani = animation.FuncAnimation(fig, animate, interval = 10, repeat=False)
#ani.save('moving_points.mp4', writer='ffmpeg')
#clip = VideoClip(animate, duration=tmax)
#clip.write_gif('lagrangian.gif', fps=24)
plt.show()