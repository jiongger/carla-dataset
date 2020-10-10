import numpy as np
import os

class carla_imu(object):

    def __init__(self, file=None, array=None, label=None):
        self.frame = None
        self.time = None
        self.x, self.y, self.z = None, None, None
        self.roll, self.pitch, self.yaw = None, None, None
        self.ax, self.ay, self.az = None, None, None
        self.gx, self.gy, self.gz = None, None, None
        self.compass = None
        if file is not None:
            #print('BUILD IMU ARRAY FROM FILE')
            self.from_imu_logs(file)
        if array is not None:
            #print('BUILD IMU ARRAY FROM ARRAY')
            self.from_array(array)
        self.label = label
        self.index = -1
    
    def from_imu_logs(self, file):
        assert os.path.isfile(file)
        imu_logs = open(file, 'r').readlines()
        imu_log_len = len(imu_logs)
        assert len(imu_logs[0].rstrip('\n').split(' ')) == 15, 'cannot parse imu log %s with length %d' %(file, len(imu_logs[0]))
        imu_data = np.zeros((imu_log_len, 15), dtype=np.float)
        #print(imu_log_len)
        for i,imu_log in enumerate(imu_logs):
            #print(imu_log)
            for j,imu_m in enumerate(imu_log.rstrip('\n').split(' ')):
                imu_data[i,j] = float(imu_m)
                #print(i,j,imu_data[i,j])
        self.frame, self.time = imu_data[:, 0:2].T
        self.x, self.y, self.z = imu_data[:, 2:5].T
        self.roll, self.pitch, self.yaw = imu_data[:, 5:8].T
        self.ax, self.ay, self.az = imu_data[:, 8:11].T
        self.gx, self.gy, self.gz = imu_data[:, 11:14].T
        self.compass = imu_data[:, 14].T

    def from_array(self, array):
        array = np.asarray(array)
        if len(array.shape) == 1:
            assert len(array) == 15, 'cannot parse imu array with length %d' %len(array)
            self.frame, self.time = array[0:2]
            self.x, self.y, self.z = array[2:5]
            self.roll, self.pitch, self.yaw = array[5:8]
            self.ax, self.ay, self.az = array[8:11]
            self.gx, self.gy, self.gz = array[11:14]
            self.compass = array[14]
        elif len(array.shape) == 2:
            assert array.shape[1] == 15, 'cannot parse imu array with shape %d,%d' %(array.shape[0],array.shape[1])
            if array.shape[0] > 1:
                self.frame, self.time = array[:, 0:2].T
                self.x, self.y, self.z = array[:, 2:5].T
                self.roll, self.pitch, self.yaw = array[:, 5:8].T
                self.ax, self.ay, self.az = array[:, 8:11].T
                self.gx, self.gy, self.gz = array[:, 11:14].T
                self.compass = array[:, 14].T
            elif array.shape[0] == 1:                
                self.frame, self.time = array[0, 0:2]
                self.x, self.y, self.z = array[0, 2:5]
                self.roll, self.pitch, self.yaw = array[0, 5:8]
                self.ax, self.ay, self.az = array[0, 8:11]
                self.gx, self.gy, self.gz = array[0, 11:14]
                self.compass = array[0, 14]
        else:
            raise NotImplementedError
            
    def set_label(self, label):
        self.label = label
    
    def plot_track(self, left_handed=True):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        plt.plot(self.x, self.y)
        if left_handed:
            ax.invert_yaxis()
        plt.show()
        plt.close('all')
    
    def plot_accelerometer(self):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        plt.plot(range(len(self.frame)), self.ax, label='accelerometer_x')
        plt.plot(range(len(self.frame)), self.ay, label='accelerometer_y')
        plt.plot(range(len(self.frame)), self.az, label='accelerometer_z')
        plt.legend()
        ax.set_ylim((-20,20))
        plt.show()
        plt.close('all')
    
    def plot_gyroscope(self):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        plt.plot(range(len(self.frame)), self.gx, label='gyroscope_x')
        plt.plot(range(len(self.frame)), self.gy, label='gyroscope_y')
        plt.plot(range(len(self.frame)), self.gz, label='gyroscope_z')
        plt.legend()
        ax.set_ylim((-0.5,0.5))
        plt.show()
        plt.close('all')
    
    def __len__(self):
        if hasattr(self.frame, '__len__'):
            return len(self.frame)
        else:
            return 1
    def __str__(self):
        if self.label is not None:
            return self.label
        else:
            if hasattr(self.frame, '__len__'):
                return ('carla imu data array with shape %dx15:\n' %len(self)) +\
                    ('    %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' %(
                        self.frame[0], self.time[0], self.x[0], self.y[0], self.z[0], self.roll[0], self.pitch[0], self.yaw[0],
                        self.ax[0], self.ay[0], self.az[0], self.gx[0], self.gy[0], self.gz[0], self.compass[0])) +\
                    ('    ... ...\n' if len(self) > 2 else '') +\
                    ('    %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' %(
                        self.frame[-1], self.time[-1], self.x[-1], self.y[-1], self.z[-1], self.roll[-1], self.pitch[-1], self.yaw[0-1],
                        self.ax[-1], self.ay[-1], self.az[-1], self.gx[-1], self.gy[-1], self.gz[-1], self.compass[-1]) if len(self) > 1 else '')
            else:
                return 'carla imu data record:\n %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n' %(
                        self.frame, self.time, self.x, self.y, self.z, self.roll, self.pitch, self.yaw,
                        self.ax, self.ay, self.az, self.gx, self.gy, self.gz, self.compass)
    
    def __getitem__(self, frame):
        if isinstance(frame, int):
            index = np.where(self.frame == frame)
            assert len(index) == 1
            return carla_imu(
                array=np.asarray([
                                  self.frame[index], self.time[index],
                                  self.x[index], self.y[index], self.z[index],
                                  self.roll[index], self.pitch[index], self.yaw[index],
                                  self.ax[index], self.ay[index], self.az[index],
                                  self.gx[index], self.gy[index], self.gz[index],
                                  self.compass[index]
                                 ]).T
                )
        elif isinstance(frame, slice) or hasattr(frame, '__iter__'):
            if isinstance(frame, slice):
                frame_list = np.arange(frame.start, frame.stop, frame.step)
            else:
                frame_list = np.asarray(frame)
            index = np.where(self.frame == frame_list[0])[0]
            for frame in frame_list[1:]:
                assert len(np.where(self.frame == frame)) == 1
                #print(np.where(self.frame == frame)[0])
                index = np.append(index, np.where(self.frame == frame)[0])
            #print(frame_list)
            #print(index, type(index))
            index = (index)
            return carla_imu(
                array=np.vstack([
                                 self.frame[index], self.time[index],
                                 self.x[index], self.y[index], self.z[index],
                                 self.roll[index], self.pitch[index], self.yaw[index],
                                 self.ax[index], self.ay[index], self.az[index],
                                 self.gx[index], self.gy[index], self.gz[index],
                                 self.compass[index]
                                ]).T
                )
        else:
            raise IndexError
    def get_keys(self):
        return self.frame
    
    def __iter__(self):
        return self
    def __next__(self):
        self.index += 1
        if self.index >= len(self.frame):
            raise StopIteration
        return carla_imu(
            array=np.asarray([
                             self.frame[self.index], self.time[self.index],
                             self.x[self.index], self.y[self.index], self.z[self.index],
                             self.roll[self.index], self.pitch[self.index], self.yaw[self.index],
                             self.ax[self.index], self.ay[self.index], self.az[self.index],
                             self.gx[self.index], self.gy[self.index], self.gz[self.index],
                             self.compass[self.index]
                            ])
        )
    
    def copy(self):
        return carla_imu(
            array=np.vstack([
                             self.frame,self.time,
                             self.x,self.y,self.z,
                             self.roll,self.pitch,self.yaw,
                             self.ax,self.ay,self.az,
                             self.gx,self.gy,self.gz,
                             self.compass
                             ]).T
            )
    
    def extend(self, extension):
        if isinstance(extension, carla_imu):
            pass
        elif hasattr(extension, '__len__'):
            extension = carla_imu(np.asarray(extension))
        else:
            raise NotImplementedError
        self.frame = np.append(self.frame, extension.frame)
        self.time = np.append(self.time, extension.time)
        self.x = np.append(self.x, extension.x)
        self.y = np.append(self.y, extension.y)
        self.z = np.append(self.z, extension.z)
        self.roll = np.append(self.roll, extension.roll)
        self.pitch = np.append(self.pitch, extension.pitch)
        self.yaw = np.append(self.yaw, extension.yaw)
        self.ax = np.append(self.ax, extension.ax)
        self.ay = np.append(self.ay, extension.ay)
        self.az = np.append(self.az, extension.az)
        self.gx = np.append(self.gx, extension.gx)
        self.gy = np.append(self.gy, extension.gy)
        self.gz = np.append(self.gz, extension.gz)
        self.compass = np.append(self.compass, extension.compass)


if __name__ == '__main__': # DEBUG only
    imu = carla_imu(file=r'D:\GitHub\carla-dataset\detbase_opposite\2\imu.log')
    for rec in imu:
        print(rec)
    print(imu.copy())
    print(imu[3117])
    print(imu[3117,3120,3122])
    print(imu[3117:3190])
    a = imu[3117:3118]
    print(a)
    b = imu[3120, 3122]
    a.extend(b)
    print(a)
    imu.plot_track()
    imu.plot_accelerometer()
    imu.plot_gyroscope()

def from_imu_logs(file):
    return carla_imu(file=file)

def from_array(array):
    return carla_imu(array=array)

def extend(imu, extension):
    assert isinstance(imu, carla_imu)
    extended_imu = imu.copy().extend(extension)
    return extended_imu

def copy(imu):
    assert isinstance(imu, carla_imu)
    return imu.copy()