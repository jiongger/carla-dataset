# User Interface of plot_core.py

import numpy as np
import matplotlib as plt
import argparse

import sys
import os

from lib.pointcloud import pointcloud
import lib.plot_core as plot_core

FLAGS = argparse.ArgumentParser(
    description=__doc__)
FLAGS.add_argument(
    '--plane',
    metavar='P',
    action='store_true',
    help='2d mode execution')
FLAGS.add_argument(
    'filename_list',
    metavar='F',
    type=str,
    nargs='+',
    help='list of path/to/point/cloud/file'
)
FLAGS.add_argument(
    '-l','--upper_limitation_of_number_of_points',
    metavar='L',
    type=int,
    default=90000,
    help='the max number of displayable points (int, default:90000)'
)
FLAGS.add_argument(
    '-s', '--size',
    metavar='S',
    type=float,
    default=0.01,
    help='the display size of each point(float, default:0.01)'
)
FLAGS.add_argument(
    '-m','--marker',
    metavar='M',
    type=str,
    default='.',
    help='the shape of display points(str, default:.)\nyou may consult the matplotlib document'
)
FLAGS.add_argument(
    '--show_figure',
    metavar='F',
    action='store_true',
    help='show figure'
)
FLAGS.add_argument(
    '--view',
    metavar='V',
    type=int,
    default=2,
    help='specify the perspect of view, functional when plot 2d scatter(int, default: 2)\n\t front view:0\n\t side view:1\n\t: bird-eye view:2'
)
FLAGS.add_argument(
    '--title',
    metavar='T',
    type=str,
    default='',
    help='title the scatter'
)
FLAGS.add_argument(
    '--save',
    metavar='S',
    type=str,
    default=None,
    help='[optional] path/to/save/figure'
)
FLAGS.add_argument(
    '--dpi',
    metavar='D',
    type=int,
    default=320,
    help='[optional] resolution of saved figure'
)
FLAGS.add_argument(
    '--exception',
    metavar='E',
    default='stop',
    choices=['skip', 'stop'],
    help='specify how to delta with exception(default:skip)'
)
FLAGS.add_argument(
    '--left_handed',
    metavar='L',
    action='store_true',
    help='left handed coordinate system'
)
FLAGS = FLAGS.parse_args()
FIG_CONFIGS = {
    'size': FLAGS.size,
    'marker': FLAGS.marker,
    'title': FLAGS.title,
    'dpi': FLAGS.dpi,
    'left_handed': FLAGS.left_handed
}



def main():

    pc_list = []
    number_of_points = 0
    for filename in FLAGS.filename_list:
        if os.path.exists(filename):
            pass
        else:
            if FLAGS.exception == 'skip':
                print('ERROR: cannot open file %s, skipped.' %filename)
            else:
                print('ERROR: cannot open file %s, aborted.' %filename)
                sys.exit()
        pc = pointcloud(filename)
        number_of_points = number_of_points + pc.number_of_points
        pc_list.append(pc)
    
    if number_of_points > FLAGS.upper_limitation_of_number_of_points:
        for pc in pc_list:
            pc.downsampling('fixed-step', FLAGS.upper_limitation_of_number_of_points/number_of_points)

    if FLAGS.plane:
        plot_core.plot_2d(pc_list, FLAGS.view, save=FLAGS.save, show_figure=FLAGS.show_figure, args=FIG_CONFIGS)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()
    