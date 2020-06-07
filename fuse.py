from pointcloud import pointcloud

import argparse
import os
import sys

def main():
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
        help='path/to/save/results (str, default:consistent with path)'
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
    args = argparser.parse_args()
    if args.coperception_vehicles_list is None:
        args.coperception_vehicles_list = range(1,args.number_of_coperception_vehicles+1)
    if args.save_results_to is None:
        args.save_results_to = args.path
    assert args.master in args.coperception_vehicles_list
    MASTER_INDEX = args.coperception_vehicles_list.index(args.master)
    assert args.number_of_coperception_vehicles == len(args.coperception_vehicles_list)
    assert (args.order == 'T' and args.number_of_coperception_vehicles >=2) or (args.order == 'V')

    pc_file_list = []
    total_file_count = 0
    for index in args.coperception_vehicles_list:
        ego_pc_file_list = []
        count = 0
        for filestr in os.listdir(args.path):
            if str.find(filestr, '.txt') >= 0:
                if str.find(filestr, '%s%d_%s' %(args.prefix,index,args.suffix)) >= 0:
                    ego_pc_file_list.append(os.path.join(args.path, filestr))
                    count = count + 1
                    total_file_count = total_file_count + 1
        pc_file_list.append(ego_pc_file_list)
        print('retrieved %d shots @ vehicle %d' %(count, index))
    print('retrieved %d shots from %d vehicles in total' %(total_file_count, args.number_of_coperception_vehicles))
    
    if args.order == 'T':
        for i in range(len(pc_file_list[MASTER_INDEX])):
            print('fusing @ timestamp %d, retrieving from %d vehicles...' %(i+1,len(pc_file_list)))
            master_pc = pointcloud(pc_file_list[MASTER_INDEX][i])
            if args.coordinate == 'G':
                master_pc.rotation(master_pc.orientation)
                master_pc.translation(master_pc.location)
            for index, assist_pc_file in enumerate(pc_file_list):
                if index == MASTER_INDEX:
                    continue
                assist_pc = pointcloud(assist_pc_file[i])
                if args.coordinate == 'G':
                    assist_pc.rotation(assist_pc.orientation)
                    assist_pc.translation(assist_pc.location)
                else:
                    assist_pc.rotation(assist_pc.orientation - master_pc.orientation)
                    assist_pc.translation(assist_pc.location - master_pc.location)
                master_pc.merge(assist_pc)
            master_pc.save_to_disk(os.path.join(args.save_results_to, 'time%d.txt' %(i+1)), True)
    
    if args.order == 'V':
        for i,ego in enumerate(args.coperception_vehicles_list):
            print('fusing @ vehicle %d, retrieving from %d shots...' %(ego, len(pc_file_list[i])))
            master_pc = pointcloud(pc_file_list[i][0])
            if args.coordinate == 'G':
                master_pc.rotation(master_pc.orientation)
                master_pc.translation(master_pc.location)
            for assist_pc_file in pc_file_list[i][1:]:
                assist_pc = pointcloud(assist_pc_file)
                if args.coordinate == 'G':
                    assist_pc.rotation(assist_pc.orientation)
                    assist_pc.translation(assist_pc.location)
                else:
                    assist_pc.rotation(assist_pc.orientation - master_pc.orientation)
                    assist_pc.translation(assist_pc.location - master_pc.location)
                master_pc.merge(assist_pc)
            master_pc.save_to_disk(os.path.join(args.save_results_to, 'ego%d.txt' %ego), True)
    


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\naborted\n')
        pass
    finally:
        print('\ndone\n')

