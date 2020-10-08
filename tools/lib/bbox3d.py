class bbox3d(object):

    def __init__(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None, label=None):
        self.h = h if h is not None else 0
        self.w = w if w is not None else 0
        self.l = l if l is not None else 0
        self.x = x if x is not None else 0
        self.y = y if y is not None else 0
        self.z = z if z is not None else 0
        self.ry = ry if ry is not None else 0
        self.label = label
    
    def set_value(self, x=None, y=None, z=None, h=None, w=None, l=None, ry=None):        
        if h is not None: self.h = h 
        if w is not None: self.w = w 
        if l is not None: self.l = l 
        if x is not None: self.x = x 
        if y is not None: self.y = y 
        if z is not None: self.z = z 
        if ry is not None: self.ry = ry
    def set_label(self, label):
        self.label = label
    
    def flatten(self, axis):
        assert axis==0 or axis==1 or axis==2
        import numpy as np
        from math import sin, cos, pi
        if axis == 0:
            v1 = np.array([self.y-self.w/2, self.z+self.h/2])
            v2 = np.array([self.y-self.w/2, self.z-self.h/2])
            v3 = np.array([self.y+self.w/2, self.z-self.h/2])
            v4 = np.array([self.y+self.w/2, self.z+self.h/2])
        if axis == 1:
            v1 = np.array([self.x-self.l/2, self.z+self.h/2])
            v2 = np.array([self.x-self.l/2, self.z-self.h/2])
            v3 = np.array([self.x+self.l/2, self.z-self.h/2])
            v4 = np.array([self.x+self.l/2, self.z+self.h/2])
        if axis == 2:
            v1 = np.array([self.x-self.l/2*cos(self.ry)-self.w/2*sin(self.ry),
                            self.y-self.l/2*sin(self.ry)+self.w/2*cos(self.ry)])
            v2 = np.array([self.x-self.l/2*cos(self.ry)+self.w/2*sin(self.ry),
                            self.y-self.l/2*sin(self.ry)-self.w/2*cos(self.ry)])
            v3 = np.array([self.x+self.l/2*cos(self.ry)+self.w/2*sin(self.ry),
                            self.y+self.l/2*sin(self.ry)-self.w/2*cos(self.ry)])
            v4 = np.array([self.x+self.l/2*cos(self.ry)-self.w/2*sin(self.ry),
                            self.y+self.l/2*sin(self.ry)+self.w/2*cos(self.ry)])
        return v1,v2,v3,v4
    
    def from_kitti_annotation(self, label):
        infos = label.rstrip('\n').split(' ')
        self.set_value(x=float(infos[11]), y=float(infos[12]), z=float(infos[13]), 
                            h=float(infos[8]), w=float(infos[9]), l=float(infos[10]), ry=float(infos[14]))

    def from_simplified_annotation(self, label):
        infos = label.rstrip('\n').split(' ')
        self.set_value(x=float(infos[5]), y=float(infos[6]), z=float(infos[7]), 
                            h=float(infos[2]), w=float(infos[3]), l=float(infos[4]), ry=float(infos[8]))

    def copy(self):
        return self.__init__(
            self.x, self.y, self.z,
            self.h, self.w, self.l,
            self.ry, self.label
        )
    
    def __str__(self):
        return self.label if self.label is not None \
            else 'bbox3d: (x=%.2f,y=%.2f,z=%.2f,h=%.2f,w=%.2f,l=%.2f,ry=%.2f)' \
                    %(self.x,self.y,self.z,self.h,self.w,self.l,self.ry)

def from_kitti_annotation(label):
    infos = label.rstrip('\n').split(' ')
    bbox = bbox3d()
    bbox.set_value(x=float(infos[11]), y=float(infos[12]), z=float(infos[13]), 
                        h=float(infos[8]), w=float(infos[9]), l=float(infos[10]), ry=float(infos[14]))
    return bbox

def from_simplified_annotation(label):
    infos = label.rstrip('\n').split(' ')
    bbox = bbox3d()
    bbox.set_value(x=float(infos[5]), y=float(infos[6]), z=float(infos[7]), 
                        h=float(infos[2]), w=float(infos[3]), l=float(infos[4]), ry=float(infos[8]))
    return bbox
