import os
import numpy as np
import torch

import lib.roipool3d.roipool3d_utils as roipool3d_utils
import lib.object3d as object3d
import lib.bbox3d as bbox3d

def clean_up(BASE_DIR, SPLIT, skip=30):
    # remove unnecessary label files
    VELO_DIR = os.path.join(BASE_DIR, 'object', SPLIT, 'velodyne')
    LABEL_DIR = os.path.join(BASE_DIR, 'object', SPLIT, 'label_2')
    IMAGE_DIR = os.path.join(BASE_DIR, 'object', SPLIT, 'image_2')
    IMU_FILE = os.path.join(BASE_DIR, 'imu.log')
    TMP_FILE = os.path.join(BASE_DIR, 'imu.tmp')
    if os.path.exists(os.path.join(BASE_DIR, 'ImageSets')):
        import shutil
        shutil.rmtree(os.path.join(BASE_DIR, 'ImageSets'))
    os.makedirs(os.path.join(BASE_DIR, 'ImageSets'))

    index_file = open(os.path.join(BASE_DIR, 'ImageSets', SPLIT+'.txt'), 'w')
    label_list = os.listdir(LABEL_DIR).copy()
    label_list.sort()
    imu = open(IMU_FILE, 'r')
    log = open(TMP_FILE, 'w')
    for index,label_name in enumerate(label_list):
        velo_name = label_name[:-4] + '.bin'
        img_name = label_name[:-4] + '.png'
        print(index, label_name)
        log = imu.readline()
        if index<skip: # clean preliminatry files
            os.remove(os.path.join(LABEL_DIR, label_name))
            if os.path.exists(os.path.join(VELO_DIR, velo_name)) and os.path.exists(os.path.join(IMAGE_DIR, img_name)):
                os.remove(os.path.join(VELO_DIR, velo_name))
                os.remove(os.path.join(IMAGE_DIR, img_name))
            else:
                raise IndexError
            continue
        if os.path.exists(os.path.join(VELO_DIR, velo_name)) and os.path.exists(os.path.join(IMAGE_DIR, img_name)):
            print(label_name[:-4], file=index_file)
            print(log.rstrip('\n'), file=TMP_FILE)
        else:
            raise IndexError

    index_file.close()
    imu.close()
    log.close()
    os.remove(IMU_FILE)
    os.rename(TMP_FILE, IMU_FILE)


def refinement(BASE_DIR, SPLIT, MINIMUM_CLUSTER_SIZE=15):
    # exclude empty(non-visible) boxes
    VELO_DIR = os.path.join(BASE_DIR, 'object', SPLIT, 'velodyne')
    LABEL_DIR = os.path.join(BASE_DIR, 'object', SPLIT, 'label_2')

    for label_name in os.listdir(LABEL_DIR):
        assert os.path.exists(os.path.join(LABEL_DIR, label_name))
        print('file=%s' %label_name)
        label_file = open(os.path.join(LABEL_DIR, label_name), 'r')
        tmp_label_file = open(os.path.join(LABEL_DIR, label_name[:-4]+'.tmp'), 'w')
        velo_file = os.path.join(VELO_DIR, label_name[:-4]+'.bin')

        labels = label_file.readlines()
        gt_boxes3d = np.zeros((len(labels),7), dtype=np.float32)
        for i,label in enumerate(labels):
            gt_box3d = object3d.Object3d(label)
            gt_boxes3d[i, 0:3] = gt_box3d.pos
            gt_boxes3d[i, 3], gt_boxes3d[i, 4], gt_boxes3d[i, 5] = gt_box3d.h, gt_box3d.w, gt_box3d.l
            gt_boxes3d[i, 6] = gt_box3d.ry
        
        bbox_list = []
        bbox_list_all = []

        velo = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        def trans_lidar_to_cam(pts_lidar):
            pts_rect = pts_lidar.copy()
            for pt_rect in pts_rect:
                pt_rect[0], pt_rect[1], pt_rect[2] = -pt_rect[1], -pt_rect[2], pt_rect[0]
            return pts_rect
        velo_xyz = trans_lidar_to_cam(velo[:, 0:3])

        boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(velo_xyz), torch.from_numpy(gt_boxes3d))
        assert len(boxes_pts_mask_list) == len(labels)
        for k in range(boxes_pts_mask_list.__len__()):
            pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
            pts_in_box = velo_xyz[pt_mask_flag]
            bbox = bbox3d.from_kitti_annotation(labels[k])
            bbox.set_label(str(len(pts_in_box)))
            if -120<=bbox.x<=120 and -120<=bbox.z<=120: # all boxes should in sensor range
                bbox_list_all.append(bbox)
                #print(pts_in_box.shape)
                if len(pts_in_box) < MINIMUM_CLUSTER_SIZE: # ignore tiny clusters
                    continue
                else:
                    bbox = bbox3d.from_kitti_annotation(labels[k])
                    print(labels[k].rstrip('\n'), file=tmp_label_file)
                    bbox_list.append(bbox)
        
        # TODO: re-generate truncated & occluded (optional)

        label_file.close()
        tmp_label_file.close()
        os.remove(os.path.join(LABEL_DIR, label_name))
        os.rename(os.path.join(LABEL_DIR, label_name[:-4]+'.tmp'), os.path.join(LABEL_DIR, label_name))

        bev_file1 = os.path.join(VELO_DIR, label_name[:-4]+'_all.png')
        bev_file2 = os.path.join(VELO_DIR, label_name[:-4]+'.png')
        bev_file3 = os.path.join(VELO_DIR, label_name[:-4]+'_side_all.png')
        from lib.plot_core import plot_2d_with_bbox
        from lib.pointcloud import pointcloud
        from math import pi
        pc = pointcloud(velo_file, use_intensity=True)
        for bbox in bbox_list_all:
            bbox.x, bbox.y, bbox.z = bbox.z, -bbox.x, -bbox.y
            bbox.ry = -(bbox.ry + pi/2)
        for bbox in bbox_list:
            bbox.x, bbox.y, bbox.z = bbox.z, -bbox.x, -bbox.y
            bbox.ry = -(bbox.ry + pi/2)
        plot_2d_with_bbox(pc, view=2, bbox_list=bbox_list_all, save=bev_file1, show_figure=False, args={'line_color':'red', 'xmin':-120, 'xmax':120, 'ymin':-120, 'ymax':120})
        plot_2d_with_bbox(pc, view=2, bbox_list=bbox_list, save=bev_file2, show_figure=False, args={'line_color':'blue', 'xmin':-120, 'xmax':120, 'ymin':-120, 'ymax':120})
        plot_2d_with_bbox(pc, view=1, bbox_list=bbox_list_all, save=bev_file3, show_figure=False, args={'line_color':'red', 'xmin':-120, 'xmax':120, 'ymin':-10, 'ymax':10})


def build_kitti_structure(PATH, SPLIT, LIDAR, CAMERA):
    VELO_DIR = os.path.join(PATH, 'object', SPLIT, 'velodyne') if LIDAR else None
    IMAGE_DIR = os.path.join(PATH, 'object', SPLIT, 'image_2') if CAMERA else None
    LABEL_DIR = os.path.join(PATH, 'object', SPLIT, 'label_2') if LIDAR else None
    CALIB_DIR = os.path.join(PATH, 'object', SPLIT, 'calib') if CAMERA else None
    import shutil
    # create empty folders
    if CAMERA:
        if os.path.exists(CALIB_DIR):
            shutil.rmtree(CALIB_DIR)
        os.makedirs(CALIB_DIR)
        if os.path.exists(IMAGE_DIR):
            shutil.rmtree(IMAGE_DIR)
        os.makedirs(IMAGE_DIR)
    if LIDAR:
        if os.path.exists(VELO_DIR):
            shutil.rmtree(VELO_DIR)
        os.makedirs(VELO_DIR)
        if os.path.exists(LABEL_DIR):
            shutil.rmtree(LABEL_DIR)
        os.makedirs(LABEL_DIR)
    return {'velo_dir':VELO_DIR, 'label_dir':LABEL_DIR, 'image_dir':IMAGE_DIR, 'calib_dir':CALIB_DIR}


def ego_imu_callback(imu, ego_imu_log):
    print(imu.frame, imu.timestamp, 
            imu.transform.location.x, imu.transform.location.y, imu.transform.location.z, 
            imu.transform.rotation.roll, imu.transform.rotation.pitch, imu.transform.rotation.yaw, 
            imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z, 
            imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z, 
            imu.compass, file=ego_imu_log)

def ego_gnss_callback(gnss, ego_gnss_log):
    print(gnss.frame, gnss.timestamp, 
                gnss.transform.location.x, gnss.transform.location.y, gnss.transform.location.z,
                gnss.transform.rotation.roll, gnss.transform.rotation.pitch, gnss.transform.rotation.yaw,
                gnss.latitude, gnss.longitude, gnss.altitude, file=ego_gnss_log)
