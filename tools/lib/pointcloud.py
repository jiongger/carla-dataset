import numpy as np
from random import randint, sample

from math import sin, cos, tan, pi

class pointcloud:

    def __init__(self, filename=None, skip=0, use_intensity=False, use_rgb=False):        
        if filename is not None:
            self.load(filename, skip, use_intensity, use_rgb)
        else:
            self.frame=-1
            self.timestamp=-1.0
            self.number_of_points=-1
            self.use_intensity=False
            self.use_rgb=False
    

    def load(self, filename, skip, use_intensity, use_rgb):
        assert isinstance(filename, str)

        if filename[-3:] == 'txt':
            infile = open(filename, 'r')            
            raw_data = infile.readlines()
            raw_cloud = raw_data[skip:]

            # point := x,y,z, I, R,G,B
            points = []
            for line in raw_cloud:
                point = [float(x) for x in line.rstrip('\n').split(' ')]
                assert len(point) == 3+int(use_intensity)+3*int(use_rgb), (len(point),3+int(use_intensity)+3*int(use_rgb))
                points.append(point)
            self.cloud = np.asarray(points)
        
        elif filename[-3:] == 'bin':
            points = np.fromfile(filename, dtype=np.float32)
            assert len(points) % (3+int(use_intensity)+3*int(use_rgb)) == 0, (len(points),(3+int(use_intensity)+3*int(use_rgb)))
            points = np.reshape(points, (-1, 3+int(use_intensity)+3*int(use_rgb)))
            self.cloud = points.copy()

        else:
            print('\nunable to parse file\n')
            raise NotImplementedError
    
        self.number_of_points = len(points)
        self.use_intensity = use_intensity
        self.use_rgb = use_rgb
    

    def translation(self, diff): # diff:=(diff_x, diff_y, diff_z) @ any array-like objects
        assert self.number_of_points >=0
        assert len(diff) == 3
        diff = np.asarray(diff)
        if self.use_rgb or self.use_intensity:
            diff = np.hstack((diff, np.zeros(3*int(self.use_rgb)+int(self.use_intensity)))) # extend diff to I,RGB dimensions
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
        # set rotation matrix for rotations @ (roll, pitch, yaw)
        rota = rota.dot(rotz)
        rota = rota.dot(roty)
        rota = rota.dot(rotx)

        rot_cloud = []
        for point in self.cloud:
            pse_point = np.dot(rota, point[0:3].transpose()) # throw RGB
            if self.use_intensity or self.use_rgb:
                pse_point = np.hstack((pse_point.transpose(), point[3:])) # retrieve I,RGB dimension
            rot_cloud.append(pse_point)
        rot_cloud = np.asarray(rot_cloud)
        self.cloud = rot_cloud.copy()


    def merge(self, pc): # merge another point cloud into current point cloud
        # remember to transform both point cloud to one consistent coordinate system first
        # via rotation() and translation() method
        assert isinstance(pc, pointcloud)
        assert self.number_of_points >= 0
        assert pc.number_of_points >= 0
        assert pc.use_rgb == self.use_rgb
        assert pc.use_intensity == self.use_intensity

        self.cloud = np.concatenate([self.cloud, pc.cloud])
        self.number_of_points = len(self.cloud)
    

    def shade(self, color=None):
        assert self.number_of_points >= 0
        if color is None:
            color = [randint(0,7)*32/255, randint(0,7)*32/255, randint(0,7)*32/255]
        elif isinstance(color, int):
            assert color>=0 and color<=255
            color = [color/255, color/255, color/255]
        elif isinstance(color, float):
            assert color>=0 and color<=1
            color = [color, color, color]
        elif hasattr(color, 'len'):
            assert len(color) == 3
        else:
            print('\ninvaild color\n')
            raise NotImplementedError
        color = np.asarray(color)

        shaded_cloud = []
        for point in self.cloud:
            pse_point = np.hstack((point[0:3+int(self.use_intensity)], color))
            shaded_cloud.append(pse_point)
        shaded_cloud = np.asarray(shaded_cloud)
        self.cloud = shaded_cloud.copy()
        self.use_rgb = True


    def reflect(self, intensity):
        if hasattr(intensity, 'len'):
            assert len(intensity) == self.number_of_points
            pse_points = self.cloud[:,0:3]
            pse_points = np.hstack((pse_points, intensity))
            if self.use_rgb:
                pse_points = np.hstack((pse_points, self.cloud[-3:]))
        elif isinstance(intensity, float):
            assert intensity>=0 and intensity<=1
            pse_points = []
            for point in self.cloud:
                pse_point = point[0:3]
                pse_point = np.hstack((pse_point, [intensity]))
                if self.use_rgb:
                    pse_point = np.hstack((pse_point, point[-3:]))
                pse_points.append(pse_point)
        else:
            print('\ninvaild intensity\n')
            raise NotImplementedError
        
        self.cloud = pse_points.copy()
        self.use_intensity = True


    def copy(self): # get a copy of current point cloud
        pc = pointcloud()
        pc.cloud = self.cloud.copy()
        pc.number_of_points = self.number_of_points
        pc.use_rgb = self.use_rgb
        pc.use_intensity = self.use_intensity
        return pc


    def inverse(self, axis):
        if isinstance(axis, str):
            if axis == 'x':
                axis = 0 
            elif axis == 'y':
                axis = 1
            elif axis == 'z':
                axis = 2
        if not isinstance(axis, int):
            print('\ninvaild axis\n')
            raise NotImplementedError
        assert axis>=0 and axis<=2

        inversed_cloud = []
        for point in self.cloud:
            pse_point = point.copy()
            pse_point[axis] = -pse_point[axis]
            inversed_cloud.append(pse_point)
        inversed_cloud = np.asarray(inversed_cloud)
        self.cloud = inversed_cloud.copy()


    def save_to_disk(self, filename, _format=None, export_intensity=None, export_rgb=None):
        assert self.number_of_points >= 0
        assert isinstance(filename, str)
        if export_intensity is None:
            export_intensity = self.use_intensity
        if export_rgb is None:
            export_rgb = self.use_rgb
        if export_intensity:
            assert self.use_intensity
        if export_rgb:
            assert self.use_rgb
        if _format is None:
            _format = filename[-3:]

        if _format == 'txt':
            save_file = open(filename, 'w')
            for point in self.cloud:
                print('%f %f %f' %(point[0], point[1], point[2]), end='', file=save_file) # location:= x,y,z
                if export_intensity:
                    print(' %f' %point[3], end='', file=save_file)
                if export_rgb:
                    print(' %f %f %f' %(point[-3], point[-2], point[-1]), end='', file=save_file)
                print(file=save_file)
        elif _format == 'bin':
            import struct
            save_file = open(filename, 'wb')
            for point in self.cloud:
                save_file.write(struct.pack('fff', point[0], point[1], point[2]))
                if export_intensity:
                    save_file.write(struct.pack('f', point[3]))
                if export_rgb:
                    save_file.write(struct.pack('fff', point[-3], point[-2], point[-1]))
        else:
            print('\ninvaild format\n')
            raise NotImplementedError


    def downsampling(self, mode, ratio):
        assert self.number_of_points >= 0
        assert mode == 'random' or mode == 'fixed-step'
        if mode == 'random':
            self.cloud = sample(self.cloud, int(self.number_of_points*ratio))
        elif mode == 'fixed-step':
            self.cloud = self.cloud[::int(1/ratio)]
        self.number_of_points = len(self.cloud)
