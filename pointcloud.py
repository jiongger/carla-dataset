import numpy as np
from random import randint, sample

from math import sin, cos, tan, pi

class pointcloud:

    def __init__(self, infile=None, color=None, left_handed_flag=True):        
        if infile is not None:
            self.load(infile, color, left_handed_flag)
        else:
            self.frame=-1
            self.timestamp=-1.0
            self.merged_flag=False
            self.number_of_points=-1
            self.has_a_head=False
    

    def load(self, infile, color, left_handed_flag):
        if isinstance(infile, str): # in case you give a file name instead of a file object
            infile = open(infile, 'r')
            
        raw_data = infile.readlines()
        raw_header = raw_data[0]
        raw_cloud = raw_data[1:]
        infos = [float(x) for x in raw_header.rstrip('\n').split(' ')]
        if color is None:
            color = [randint(0,7)*32, randint(0,7)*32, randint(0,7)*32]
        elif isinstance(color, int):
            color = [color, color, color]
        assert len(color) == 3
        color = np.asarray(color)

        if len(infos) == 10: # parse the header
            self.frame = infos[0]
            self.timestamp = infos[1]
            self.location = np.asarray((infos[2], infos[3], -infos[4] if left_handed_flag else infos[4]))
            self.orientation = np.asarray((infos[5], infos[6], infos[7]+90 if left_handed_flag else infos[7]))
            self.horizontal_angle = infos[8]
            self.channels = infos[9]
            self.has_a_head = True
        else:
            point = [float(x) for x in raw_cloud[0].rstrip('\n').split(' ')] # check if it's a invaild header
            if len(point) != len(infos):
                print('\nWARNING: the point cloud file have a invaild header, skipped\n')
            # the point cloud file do not have a header
            print('\nWARNING: the point cloud file do not have a header\n')
            self.frame = -1
            self.timestamp = -1.0
            self.location = np.asarray((0,0,0))
            self.orientation = np.asarray((0,0,0))
            self.horizontal_angle = -1.0
            self.channels = -1
            self.has_a_head = False
            raw_cloud = raw_data

        points = []
        for line in raw_cloud:
            point = [float(x) for x in line.rstrip('\n').split(' ')]
            if left_handed_flag:
                point[2] = -point[2]
            if len(point) == 3:
                point.extend(color)
            assert len(point) == 6
            points.append(point)
        self.cloud = np.asarray(points)
    
        self.number_of_points = len(points)
        self.merged_flag = False
        self.left_handed_flag = left_handed_flag
    

    def translation(self, diff): # diff:=(diff_x, diff_y, diff_z) @ any array-like objects
        assert self.number_of_points >=0
        assert len(diff) == 3
        diff = np.asarray(diff)
        diff = np.hstack((diff, np.zeros(3))) # extend diff to RGB dimensions
        self.cloud = self.cloud + diff
    

    def rotation(self, diff): # diff:=(roll, pitch, yaw) @ any array-like objects
    # (!important!) all roll, pitch, yaw should be measured in degrees instead of rads
        assert self.number_of_points >= 0
        assert len(diff) == 3
        diff = np.asarray(diff)

        rotx = np.zeros((3,3))
        roty = np.zeros((3,3))
        rotz = np.zeros((3,3))
        rota = np.eye(3)

        if self.left_handed_flag:
        # set rotation matrix for rotations around x-axis
            rotx[0,0] = 1
            rotx[1,1] = cos(diff[0]/180*pi)
            rotx[1,2] = -sin(diff[0]/180*pi)
            rotx[2,1] = sin(diff[0]/180*pi)
            rotx[2,2] = cos(diff[0]/180*pi)
        # set rotation matrix for rotations around y-axis
            roty[0,0] = cos(diff[1]/180*pi)
            roty[0,2] = sin(diff[1]/180*pi)
            roty[1,1] = 1
            roty[2,0] = -sin(diff[1]/180*pi)
            roty[2,2] = cos(diff[1]/180*pi)
        # set rotation matrix for rotations around z-axis
            rotz[0,0] = cos(diff[2]/180*pi)
            rotz[0,1] = -sin(diff[2]/180*pi)
            rotz[1,0] = sin(diff[2]/180*pi)
            rotz[1,1] = cos(diff[2]/180*pi)
            rotz[2,2] = 1
        else:
        # set rotation matrix for rotations around x-axis
            rotx[0,0] = 1
            rotx[1,1] = cos(diff[0]/180*pi)
            rotx[1,2] = sin(diff[0]/180*pi)
            rotx[2,1] = -sin(diff[0]/180*pi)
            rotx[2,2] = cos(diff[0]/180*pi)
        # set rotation matrix for rotations around y-axis
            roty[0,0] = cos(diff[1]/180*pi)
            roty[0,2] = -sin(diff[1]/180*pi)
            roty[1,1] = 1
            roty[2,0] = sin(diff[1]/180*pi)
            roty[2,2] = cos(diff[1]/180*pi)
        # set rotation matrix for rotations around z-axis
            rotz[0,0] = cos(-diff[2]/180*pi)
            rotz[0,1] = sin(-diff[2]/180*pi)
            rotz[1,0] = -sin(-diff[2]/180*pi)
            rotz[1,1] = cos(-diff[2]/180*pi)
            rotz[2,2] = 1
        # set rotation matrix for rotations @ (roll, pitch, yaw)
        rota = rota.dot(rotz)
        rota = rota.dot(roty)
        rota = rota.dot(rotx)

        rot_cloud = []
        for point in self.cloud:
            pse_point = np.dot(rota, point[0:3].transpose()) # throw RGB
            pse_point = np.hstack((pse_point.transpose(), point[3:6])) # retrieve RGB dimension
            rot_cloud.append(pse_point)
        rot_cloud = np.asarray(rot_cloud)
        self.cloud = rot_cloud.copy()


    def merge(self, pc): # merge another point cloud into current point cloud
        # remember to transform both point cloud to one consistent coordinate system first
        # via rotation() and translation() method
        assert isinstance(pc, pointcloud)
        assert self.number_of_points >= 0
        assert pc.number_of_points >= 0

        self.cloud = np.concatenate([self.cloud, pc.cloud])
        self.merged_flag = True
        self.number_of_points = len(self.cloud)
    
    def reshade(self, color=None):
        assert self.number_of_points >= 0
        if color is None:
            color = [randint(0,7)*32, randint(0,7)*32, randint(0,7)*32]
        elif isinstance(color, int):
            color = [color, color, color]
        assert len(color) == 3
        color = np.asarray(color)

        reshade_cloud = []
        for point in self.cloud:
            pse_point = np.hstack((point[0:3], color))
            reshade_cloud.append(pse_point)
        reshade_cloud = np.asarray(reshade_cloud)
        self.cloud = reshade_cloud.copy()
        

    def copy(self): # get a copy of current point cloud
        pc = pointcloud()
        pc.frame = self.frame
        pc.timestamp = self.timestamp
        pc.location = self.location.copy()
        pc.orientation = self.orientation.copy()
        pc.horizontal_angle = self.horizontal_angle
        pc.channels = self.channels
        pc.cloud = self.cloud.copy()
        pc.number_of_points = self.number_of_points
        pc.merged_flag = self.merged_flag
        pc.has_a_head = self.has_a_head
        pc.left_handed_flag = self.left_handed_flag
        return pc


    def save_to_disk(self, filename, points_only_flag=False, with_color_flag=True):
        assert self.number_of_points >= 0
        assert isinstance(filename, str)

        save_file = open(filename, 'w')

        if points_only_flag == False:
            if self.merged_flag == True:
                print("\nWARNING: The header of this point cloud may inconsistent according to its merged_flag.\n")
            print(self.frame, self.timestamp, 
            self.location[0], self.location[1], -self.location[2] if self.left_handed_flag else self.location[2], # location:= x,y,z
            self.orientation[0], self.orientation[1], self.orientation[2]-90 if self.left_handed_flag else self.orientation[2], # orientation:= roll,pitch,yaw
            self.horizontal_angle, self.channels, file=save_file)
        for point in self.cloud:
            if with_color_flag == False:
                print(point[0], point[1], -point[2] if self.left_handed_flag else point[2], file=save_file) # location:= x,y,z
            else:
                print(point[0], point[1], -point[2] if self.left_handed_flag else point[2], point[3], point[4], point[5], file=save_file) # locatio:= x,y,z color:=R,G,B


    def downsampling(self, mode, ratio):
        if mode == 'random':
            self.cloud = sample(self.cloud, int(self.number_of_points*ratio))
        elif mode == 'fixed-step':
            self.cloud = self.cloud[::int(1/ratio)]
        self.number_of_points = len(self.cloud)


# debug
if __name__ == '__main__':

    master_pcfile = open('ego1_lidar_measurement_4464.txt', 'r')
    assist_pcfile = open('ego2_lidar_measurement_4465.txt', 'r')
    fuse_pcfile_name = 'ego12_4466.txt'

    master_point_cloud = pointcloud(master_pcfile)
    assist_point_cloud = pointcloud(assist_pcfile)

    assist_point_cloud.rotation(assist_point_cloud.orientation - master_point_cloud.orientation)
    assist_point_cloud.translation(assist_point_cloud.location - master_point_cloud.location)

    master_point_cloud.merge(assist_point_cloud)
    master_point_cloud.save_to_disk(fuse_pcfile_name, True)
