import numpy as np
import os

class carla_imu(object):

    def __init__(self, file=None, label=None):
        self.frame = None
        self.time = None
        self.x, self.y, self.z = None, None, None
        self.roll, self.pitch, self.yaw = None, None, None
        self.ax, self.ay, self.az = None, None, None
        self.gx, self.gy, self.gz = None, None, None
        self.compass = None
        if file is not None:
            self.from_imu_logs(file)
        self.label = label
        self.index = -1
    
    def from_imu_logs(self, file):
        assert os.path.isfile(file)
        imu_logs = open(file, 'r').readlines()
        imu_log_len = len(imu_logs)
        assert len(imu_logs[0]) == 15, 'cannot parse imu log'
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
    def set_label(self, label):
        self.label = label
    
    def plot_track(self):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        plt.plot(self.x, self.y)
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
        return len(self.frame)
    def __str__(self):
        if self.label is not None:
            return self.label
        else:
            return 'carla imu data array with shape %dx15' %len(self)
    
    def __getitem__(self, frame):
        index = np.where(self.frame == frame)
        return np.asarray([
            self.frame[index], self.time[index],
            self.x[index], self.y[index], self.z[index],
            self.roll[index], self.pitch[index], self.yaw[index],
            self.ax[index], self.ay[index], self.az[index],
            self.gx[index], self.gy[index], self.gz[index],
            self.compass[index]
        ])
    
    def __iter__(self):
        return self
    def __next__(self):
        self.index += 1
        if self.index >= len(self.frame):
            raise StopIteration
        return np.asarray([
            self.frame[self.index], self.time[self.index],
            self.x[self.index], self.y[self.index], self.z[self.index],
            self.roll[self.index], self.pitch[self.index], self.yaw[self.index],
            self.ax[self.index], self.ay[self.index], self.az[self.index],
            self.gx[self.index], self.gy[self.index], self.gz[self.index],
            self.compass[self.index]
        ])


if __name__ == '__main__': # DEBUG only
    imu = carla_imu(file=r'D:\GitHub\carla-dataset\detbase_opposite\2\imu.log')
    for rec in imu:
        print(rec)
    print(imu[3117])
    print(str(imu))
    imu.plot_track()
    imu.plot_accelerometer()
    imu.plot_gyroscope()