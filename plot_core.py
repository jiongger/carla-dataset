import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pointcloud import pointcloud

def plot_3d(pc_list, size=0.01, marker='.', title='', save=None, show_figure=True, dpi=320, left_handed=False):
    raise NotImplementedError

    scatter3d = Axes3D(plt.subplots())

    for i,pc in enumerate(pc_list):
        print('ploting %d/%d...' %(i,len(pc_list)))
        x,y,z = pc.cloud[:,0], pc.cloud[:,1], pc.cloud[:,2]
        if len(pc.cloud[0]) >= 6:
            color = pc.cloud[:,-3:]
        else:
            color = None
        scatter3d.scatter(x,y,z, marker=marker, s=size, c=color)
    print('plot done %d/%d' %(len(pc_list),len(pc_list)))
    
    plt.title(title)
    if left_handed:
        ax.invert_xaxis()
    if save is not None:
        plt.savefig(str(save), dpi=dpi)
    if show_figure == True:
        plt.show()

def plot_2d(pc_list, axis=2, size=0.01, marker='.', title='', save=None, show_figure=True, dpi=320, left_handed=False):
    axis = int(axis)
    assert axis<=2 and axis>=0
    fig,ax = plt.subplots()

    for i,pc in enumerate(pc_list):
        print('ploting %d/%d...' %(i,len(pc_list)))
        x,y,z = pc.cloud[:,0], pc.cloud[:,1], pc.cloud[:,2]
        if len(pc.cloud[0]) >= 6:
            color = pc.cloud[:,-3:]
        else:
            color = None
        if axis == 0:
            plt.scatter(y,z, marker=marker, s=size, c=color)
        elif axis == 1:
            plt.scatter(x,z, marker=marker, s=size, c=color)
        elif axis == 2:
            plt.scatter(x,y, marker=marker, s=size, c=color)
    print('plot done %d/%d' %(len(pc_list),len(pc_list)))

    plt.title(title)
    if left_handed:
        ax.invert_yaxis()
    if save is not None:
        plt.savefig(str(save), dpi=dpi)
    if show_figure == True:
        plt.show()