import numpy as np

from math import sin, cos, tan, pi

class point_cloud:

    def __init__(self, file):

        self.raw_data = file.readlines()
        self.raw_header = self.raw_data[0]
        self.raw_point_cloud = self.raw_data[1:]
        infos = [float(x) for x in self.raw_header.rstrip('\n').split(' ')]
        self.frame = infos[0]
        self.timestamp = infos[1]
        self.location = np.asarray((infos[2], infos[3], infos[4]))
        self.orientation = np.asarray((infos[5], infos[6], infos[7]))
        self.horizontal_angle = infos[8]
        self.channels = infos[9]
        points = []
        for line in self.raw_point_cloud:
            point = [float(x) for x in line.rstrip('\n').split(' ')]
            points.append(point)
        self.point_cloud = np.asarray(points)
        self.number_of_point = len(points)
    

    def translation(self, diff): # diff:=(diff_x, diff_y, diff_z) @ any array-like objects
        assert len(self.point_cloud) > 0
        assert len(diff) == 3
        diff = np.asarray(diff)
        self.point_cloud = self.point_cloud - diff
    

    def rotation(self, diff): # diff:=(roll, pitch, yaw) @ any array-like objects
    # (!important!) all roll, pitch, yaw should be measured in degrees instead of rads
        assert len(self.point_cloud) > 0
        assert len(diff) == 3

        rotx = np.zeros((3,3))
        roty = np.zeros((3,3))
        rotz = np.zeros((3,3))
        rota = np.eye(3)
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
        #### carla.rotation.yaw appear to be clock-wise 
        #### while the yaw in this formula is anti-clock-wise
        rotz[0,0] = cos(-diff[2]/180*pi)
        rotz[0,1] = sin(-diff[2]/180*pi)
        rotz[1,0] = -sin(-diff[2]/180*pi)
        rotz[1,1] = cos(-diff[2]/180*pi)
        rotz[2,2] = 1
        # set rotation matrix for rotations @ (roll, pitch, yaw)
        rota = rota.dot(rotz)
        rota = rota.dot(roty)
        rota = rota.dot(rotx)

        rot_point_cloud = []
        for point in self.point_cloud:
            pse_point = np.dot(rota, point.transpose())
            rot_point_cloud.append(pse_point.transpose())
        self.point_cloud = rot_point_cloud


    def merge(self, pc): # merge another point cloud into current point cloud
        # remember to transform both point cloud to one consistent coordinate system first
        # via rotation() and translation() method
        assert isinstance(pc, point_cloud)
        assert len(self.point_cloud) > 0
        assert len(pc.point_cloud) > 0

        fused_cloud = []
        fused_raw_cloud = []
        for point, raw_point in zip(self.point_cloud, self.raw_point_cloud):
            fused_cloud.append(point)
            fused_raw_cloud.append(raw_point)
        for point, raw_point in zip(pc.point_cloud, pc.raw_point_cloud):
            fused_cloud.append(point)
            fused_raw_cloud.append(raw_point)
        self.point_cloud = np.asarray(fused_cloud)
        self.raw_point_cloud = fused_raw_cloud

    
    def save_to_disk(self, filename, points_only_flag=False):
        assert len(self.point_cloud) > 0
        assert isinstance(filename, str)

        save_file = open(filename, 'w')

        if points_only_flag == False:
            print(self.frame, self.timestamp, 
            self.location[0], self.location[1], self.location[2], # location:= x,y,z
            self.orientation[0], self.orientation[1], self.orientation[2], # orientation:= roll,pitch,yaw
            self.horizontal_angle, self.channels, file=save_file)
        for point in self.point_cloud:
            print(point[0], point[1], point[2], file=save_file) # location:= x,y,z


if __name__ == '__main__':

    master_pcfile = open('ego1_lidar_measurement_52536.txt', 'r')
    assist_pcfile = open('ego2_lidar_measurement_52537.txt', 'r')
    fuse_pcfile_name = 'ego12_52538.txt'

    ego1_lidar_loc = np.asarray([-84.76717376708984,37.042259216308594,2.0970163345336914])
    ego1_imu_rot = np.asarray([-0.0019226070726290345,1.3695826530456543,89.78825378417969])
    ego2_lidar_loc = np.asarray([-54.751708984375,-2.898881435394287,1.8585389852523804])
    ego2_imu_rot = np.asarray([0.0,0.1372799575328827,-179.70538330078125])

    master_point_cloud = point_cloud(master_pcfile)
    assist_point_cloud = point_cloud(assist_pcfile)

    assist_point_cloud.rotation(ego2_imu_rot-ego1_imu_rot)
    assist_point_cloud.translation(ego2_lidar_loc-ego1_lidar_loc)

    master_point_cloud.merge(assist_point_cloud)
    master_point_cloud.save_to_disk(fuse_pcfile_name, True)
