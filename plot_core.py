import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import singledispatch

from pointcloud import pointcloud

def plot_3d(pc_list, size=0.01, marker='.', title='', save=None, show_figure=True, dpi=320):
    scatter3d = Axes3D(plt.gcf()),

    for pc in pc_list:
        x,y,z = pc[:,0], pc[:,1], pc[:,2]
        color = pc[0,3], pc[0,4], pc[0,5]
        scatter3d.scatter(x,y,z, marker=marker, s=size, c=color)
    
    plt.title(title)
    if save is not None:
        plt.savefig(str(save), dpi=dpi)
    if show_figure == True:
        plt.show()

def plot_2d(pc_list, axis=2, size=0.01, marker='.', title='', save=None, show_figure=True, dpi=320):
    axis = int(axis)
    assert axis<=2 and axis>=0

    for i,pc in enumerate(pc_list):
        print('ploting %d/%d...' %(i,len(pc_list)))
        x,y,z = pc[:,0], pc[:,1], pc[:,2]
        color = pc[0,3], pc[0,4], pc[0,5]
        if axis == 0:
            plt.scatter(y,z, marker=marker, s=size, c=color)
        elif axis == 1:
            plt.scatter(x,z, marker=marker, s=size, c=color)
        elif axis == 2:
            plt.scatter(x,y, marker=marker, s=size, c=color)
    
    plt.title(title)
    if save is not None:
        plt.savefig(str(save), dpi=dpi)
    if show_figure == True:
        plt.show()