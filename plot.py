# User Interface of plot_core.py

import numpy as np
import matplotlib as plt
import argparse

import sys

from pointcloud import pointcloud
import plot_core

FLAGS = argparse.ArgumentParser(
    description=__doc__)
FLAGS.add_argument(
    '--mode',
    metavar='M',
    default='3d',
    choices=['2d', '3d'],
    help='choose to plot a 2d scatter or 3d scatter(default: 3d)'
)
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
    type=bool,
    default=True,
    help='choose to show figure or not(boolean, default:True)'
)
FLAGS.add_argument(
    '--axis',
    metavar='A',
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
FLAGS = FLAGS.parse_args()



def main():

    pc_list = []
    number_of_points = 0
    for filename in FLAGS.filename_list:
        try:
            pc_file = open(filename, 'r')
        except IOError:
            if FLAGS.exception == 'skip':
                print('ERROR: cannot open file %s, skipped.' %filename)
            else:
                print('ERROR: cannot open file %s, aborted.' %filename)
                sys.exit()
        pc = pointcloud(pc_file)
        number_of_points = number_of_points + pc.number_of_points
        pc_list.append(pc)
    
    if number_of_points > FLAGS.upper_limitation_of_number_of_points:
        for pc in pc_list:
            pc.downsampling('fixed-step', FLAGS.upper_limitation_of_number_of_points/number_of_points)

    if FLAGS.mode == '2d':
        plot_core.plot_2d(pc_list, FLAGS.axis, FLAGS.size, FLAGS.marker, FLAGS.title, FLAGS.save, FLAGS.show_figure, FLAGS.dpi)
    else:
        plot_core.plot_3d(pc_list, FLAGS.size, FLAGS.marker, FLAGS.title, FLAGS.save, FLAGS.show_figure, FLAGS.dpi)

if __name__ == '__main__':
    main()