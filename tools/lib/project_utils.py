import numpy as np

def get_P_rect_Mat(fov=120.0, IMAGE_X=1920, IMAGE_Y=1080, offset=(0,0,0)):
    focal_distance = IMAGE_X / ( 2.0 * np.tan( fov * np.pi/360 ))
    P_rect = np.asarray([
        [focal_distance, 0,              IMAGE_X/2, -focal_distance*offset[0]],
        [0,              focal_distance, IMAGE_Y/2, -focal_distance*offset[1]],
        [0,              0,              1,         -focal_distance*offset[2]]
    ])
    return P_rect

def get_R_rect_Mat(Cam1_transform=None, Cam0_Transform=None):
    if Cam0_Transform is None and Cam1_transform is None:
        R_rect = np.asarray([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])
    else:
        raise NotImplementedError
    return R_rect

def get_velo_to_cam_Mat(Velo_Transform=None, Cam_Transform=None):
    if Velo_Transform is None and Cam_Transform is None:
        Velo_to_Cam = np.asarray([
            [0, -1, 0,  0   ],
            [0, 0,  -1, -0.9],
            [1, 0,  0,  -0.8]
        ])
    else:
        raise NotImplementedError
    return Velo_to_Cam

def pts_to_hom(pts):
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return pts_hom

def calc_projected_pts(pts, fov=120.0, IMAGE_X=1920, IMAGE_Y=1080, Velo_Transform=None, Cam0_Transform=None, CamX_Transform=None):
    offset = (0,0,0) # TODO: calculate offset according to cam0 and camX transform
    pts_rect = np.dot(pts_to_hom(pts), np.dot(get_velo_to_cam_Mat(Velo_Transform=None, Cam_Transform=Cam0_Transform).T, get_R_rect_Mat(Cam1_transform=CamX_Transform,Cam0_Transform=Cam0_Transform).T))
    pts_img = np.dot(pts_to_hom(pts_rect), get_P_rect_Mat(fov=fov, IMAGE_X=IMAGE_X, IMAGE_Y=IMAGE_Y, offset=offset).T)
    return (pts_img[:,0:2].T / pts_img[:,2]).T

def main():
    pts = np.ones((8,3))
    print(calc_projected_pts(pts))

if __name__ == '__main__': # DEBUG only
    main()