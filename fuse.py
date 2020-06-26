from pointcloud import pointcloud

import argparse
import os
import sys
import re
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(
    description=__doc__
)
argparser.add_argument(
    'path',
    type=str,
    help='path/to/dataset'
)
argparser.add_argument(
    '-n','--number_of_coperception_vehicles',
    metavar='N',
    default=2,
    type=int,
    help='number of co-perception vehicles (int, default=2)'
)
argparser.add_argument(
    '-i', '--coperception_vehicles_list',
    metavar='I',
    default=None,
    help='[optional] index of co-perception vehicles (array-like, default=[1,2,..,N])'
)
argparser.add_argument(
    '-m', '--master',
    metavar='M',
    default=1,
    type=int,
    help='index of master vehicle (int, default=1)\ntherefore, any other vehicle are considered as assistant vehicles'
)
argparser.add_argument(
    '-o', '--order',
    metavar='O',
    default='T',
    choices=['T', 'V'],
    help='designate fuse mode, T for time-wise, V for vehicle-wise (str, default=T)'
)
argparser.add_argument(
    '-c', '--coordinate',
    metavar='C',
    default='L',
    choices=['L', 'G'],
    help='designate fuse coordinate system, L for local coordinate (from the view of master vehicle), G for global coordinate (str, default=L)'
)
argparser.add_argument(
    '-r', '--save_results_to',
    metavar='R',
    default=None,
    type=str,
    help='path/to/save/results (str, default:consistent with path)\nyou may create a non-exist directory first'
)
argparser.add_argument(
    '-p', '--prefix',
    metavar='P',
    default='ego',
    type=str,
    help=''
)
argparser.add_argument(
    '-s', '--suffix',
    metavar='S',
    default='lidar_measurement',
    type=str,
    help=''
)
FLAGS = argparser.parse_args()
if FLAGS.coperception_vehicles_list is None:
    FLAGS.coperception_vehicles_list = range(1,FLAGS.number_of_coperception_vehicles+1)
if FLAGS.save_results_to is None:
    FLAGS.save_results_to = FLAGS.path
assert FLAGS.master in FLAGS.coperception_vehicles_list
MASTER_INDEX = FLAGS.coperception_vehicles_list.index(FLAGS.master)
assert FLAGS.number_of_coperception_vehicles == len(FLAGS.coperception_vehicles_list)
assert (FLAGS.order == 'T' and FLAGS.number_of_coperception_vehicles >=2) or (FLAGS.order == 'V')


def main():
    pc_file_list = []
    total_file_count = 0
    shortest_file_sequence = -1
    for index in FLAGS.coperception_vehicles_list:
        ego_pc_file_list = []
        count = 0
        for filestr in os.listdir(FLAGS.path):
            if str.find(filestr, '.txt') >= 0:
                if str.find(filestr, '%s%d_%s' %(FLAGS.prefix,index,FLAGS.suffix)) >= 0:
                    ego_pc_file_list.append(os.path.join(FLAGS.path, filestr))
                    count = count + 1
                    total_file_count = total_file_count + 1
        ego_pc_file_list.sort( 
            key = lambda name:
                int(re.findall(r'\d+', os.path.basename(name).replace('%s%d_%s' %(FLAGS.prefix,index,FLAGS.suffix), ''))[0]) # extract timestamp in filename
        ) # execute sort to keep timestamp consistency
        pc_file_list.append(ego_pc_file_list)
        if shortest_file_sequence < 0 or shortest_file_sequence > count:
            shortest_file_sequence = count
        print('retrieved %d shots @ vehicle %d' %(count, index))
    print('retrieved %d shots from %d vehicles in total' %(total_file_count, FLAGS.number_of_coperception_vehicles))
    assert total_file_count > 0
    
    # TODO: coordinate transfer may create a mirror world 
    # TODO: figure out why?
    if FLAGS.order == 'T':
        if shortest_file_sequence < len(pc_file_list[MASTER_INDEX]):
            print('skipped %d unaligned file(s)' %(len(pc_file_list[MASTER_INDEX]) - shortest_file_sequence))
        for i in range(shortest_file_sequence):
            print('fusing @ timestamp %d, retrieving from %d vehicles...' %(i+1,len(pc_file_list)))
            master_pc = pointcloud(pc_file_list[MASTER_INDEX][i])
            for index, assist_pc_file in enumerate(pc_file_list):
                if index == MASTER_INDEX:
                    continue
                print('\tretrieving from %d/%d vehicle' %(index+2 if index<MASTER_INDEX else index+1, len(pc_file_list)))
                assist_pc = pointcloud(assist_pc_file[i])
                assist_pc.rotation(assist_pc.orientation - master_pc.orientation)
                assist_pc.translation(assist_pc.location - master_pc.location)
                master_pc.merge(assist_pc)
            if FLAGS.coordinate == 'G':
                master_pc.rotation(master_pc.orientation)
                master_pc.translation(master_pc.location)
            master_pc.save_to_disk(os.path.join(FLAGS.save_results_to, 'time%d.txt' %(i+1)), True)
    
    elif FLAGS.order == 'V':
        fig, ax = plt.subplots()
        for i,ego in enumerate(FLAGS.coperception_vehicles_list):
            trail_x = []
            trail_y = []
            print('fusing @ vehicle %d, retrieving from %d shots...' %(ego, len(pc_file_list[i])))
            master_pc = pointcloud(pc_file_list[i][0])
            print(master_pc.location, master_pc.orientation)
            trail_x.append(master_pc.location[0])
            trail_y.append(master_pc.location[1])
            for index, assist_pc_file in enumerate(pc_file_list[i][1:]):
                print('\tretrieving from %d/%d shot' %(index+2, len(pc_file_list[i])))
                assist_pc = pointcloud(assist_pc_file)
                trail_x.append(assist_pc.location[0])
                trail_y.append(assist_pc.location[1])
                assist_pc.rotation(assist_pc.orientation - master_pc.orientation)
                assist_pc.translation(assist_pc.location - master_pc.location)
                master_pc.merge(assist_pc)            
            if FLAGS.coordinate == 'G':
                master_pc.rotation(master_pc.orientation)
                master_pc.translation(master_pc.location)
            master_pc.save_to_disk(os.path.join(FLAGS.save_results_to, 'ego%d.txt' %ego), True)
            ax.plot(trail_x, trail_y, label='ego%d' %ego)
        ax.set_title('trails')
        ax.set_aspect('equal')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\naborted\n')
        pass
    finally:
        print('\ndone\n')

