import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(pc_list, save=None, show_figure=True, args=None):
    fig,ax = plt.subplots()
    ax3d = Axes3D(fig)

    if args is None:        
        args = {
            'size': 0.01,
            'marker': '.',
            'title': '',
            'dpi': 320,
            'left_handed': False
        }
    assert isinstance(args, dict)

    for i,pc in enumerate(pc_list):
        print('ploting %d/%d...' %(i,len(pc_list)))
        x,y,z = pc.cloud[:,0], pc.cloud[:,1], pc.cloud[:,2]
        if len(pc.cloud[0]) >= 6:
            color = pc.cloud[:,-3:]
        else:
            color = None
        ax3d.scatter(x,y,z, s=args['size'] if 'size' in args else 0.01, c=color,
                            marker=args['marker'] if 'marker' in args else '.')
    print('plot done %d/%d' %(len(pc_list),len(pc_list)))

    plt.title(args['title'] if 'title' in args else '')
    if 'left_handed' in args:
        if args['left_handed'] == True:
            ax.invert_yaxis()
    if save is not None:
        plt.savefig(str(save), dpi=args['dpi'] if 'dpi' in args else 320)
    if show_figure == True:
        plt.show()
    plt.close(fig)

def plot_2d(pc_list, view, save=None, show_figure=True, args=None):
    axis = int(view)
    assert axis<=2 and axis>=0
    fig,ax = plt.subplots()

    if args is None:        
        args = {
            'size': 0.01,
            'marker': '.',
            'color': 'green',
            'title': '',
            'dpi': 320,
            'left_handed': False,
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None
        }
    assert isinstance(args, dict)

    ax.set_xlim({
        'xmin': args['xmin'] if 'xmin' in args else None,
        'xmax': args['xmax'] if 'xmax' in args else None
    })
    ax.set_ylim({
        'ymin': args['ymin'] if 'ymin' in args else None,
        'ymax': args['ymax'] if 'ymax' in args else None
    })

    for i,pc in enumerate(pc_list):
        print('ploting %d/%d...' %(i,len(pc_list)))
        x,y,z = pc.cloud[:,0], pc.cloud[:,1], pc.cloud[:,2]
        if 'color' in args:
            color = args['color']
        elif len(pc.cloud[0]) >= 6:
            color = pc.cloud[:,-3:]
        else:
            color = None
        if axis == 0:
            plt.scatter(y,z, marker=args['marker'] if 'marker' in args else '.', 
                                    s=args['size'] if 'size' in args else 0.01, c=color)
        elif axis == 1:
            plt.scatter(x,z, marker=args['marker'] if 'marker' in args else '.', 
                                    s=args['size'] if 'size' in args else 0.01, c=color)
        elif axis == 2:
            plt.scatter(x,y, marker=args['marker'] if 'marker' in args else '.', 
                                    s=args['size'] if 'size' in args else 0.01, c=color)
    print('plot done %d/%d' %(len(pc_list),len(pc_list)))

    plt.title(args['title'] if 'title' in args else '')
    if 'left_handed' in args:
        if args['left_handed'] == True:
            ax.invert_yaxis()
    if save is not None:
        plt.savefig(str(save), dpi=args['dpi'] if 'dpi' in args else 320)
    if show_figure == True:
        plt.show()
    plt.close(fig)


def plot_2d_with_bbox(pc, view, bbox_list, save=None, show_figure=True, args=None):    
    axis = int(view)
    assert axis<=2 and axis>=0
    fig,ax = plt.subplots()

    if args is None:        
        args = {
            'size': 0.01,
            'marker': '.',
            'color': 'green',
            'title': '',
            'dpi': 320,
            'left_handed': False,
            'thickness': 0.5,
            'line_color': 'red',
            'xmin': None,
            'xmax': None,
            'ymin': None,
            'ymax': None
        }
    assert isinstance(args, dict)

    ax.set_xlim(
        xmin= args['xmin'] if 'xmin' in args else None,
        xmax= args['xmax'] if 'xmax' in args else None
    )
    ax.set_ylim(
        ymin= args['ymin'] if 'ymin' in args else None,
        ymax= args['ymax'] if 'ymax' in args else None
    )

    x,y,z = pc.cloud[:,0], pc.cloud[:,1], pc.cloud[:,2]
    if 'color' in args:
        color = args['color']
    elif len(pc.cloud[0]) >= 6:
        color = pc.cloud[:,-3:]
    else:
        color = 'green'
    if axis == 0:
        plt.scatter(y,z, marker=args['marker'] if 'marker' in args else '.', 
                                s=args['size'] if 'size' in args else 0.01, c=color)
    elif axis == 1:
        plt.scatter(x,z, marker=args['marker'] if 'marker' in args else '.', 
                                s=args['size'] if 'size' in args else 0.01, c=color)
    elif axis == 2:
        plt.scatter(x,y, marker=args['marker'] if 'marker' in args else '.', 
                                s=args['size'] if 'size' in args else 0.01, c=color)
    
    for bbox in bbox_list:
        bbox_v1, bbox_v2, bbox_v3, bbox_v4 = bbox.flatten(axis)
        box = plt.Polygon([bbox_v1, bbox_v2, bbox_v3, bbox_v4], fill=False, 
                                color=args['line_color'] if 'line_color' in args else 'red',
                                linewidth=args['thickness'] if 'thickness' in args else 0.5)
        ax.add_patch(box)
        if axis == 0: center=(bbox.y,bbox.z)
        elif axis == 1: center=(bbox.x,bbox.z)
        elif axis == 2: center=(bbox.x,bbox.y)
        plt.scatter(center[0], center[1], marker='*', s=0.5, color='black')
        if bbox.label is not None:
            plt.text(center[0], center[1], str(bbox.label))
    
    plt.title(args['title'] if 'title' in args else '')
    if 'left_handed' in args:
        if args['left_handed'] == True:
            ax.invert_yaxis()
    if save is not None:
        plt.savefig(str(save), dpi=args['dpi'] if 'dpi' in args else 320)
    if show_figure == True:
        plt.show()
    plt.close(fig)