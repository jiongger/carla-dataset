import cv2 as cv
import numpy as np
import os

def draw_box2d(image, x, y, w, h, color=(255,0,0), width=4, coord_norm=False):
    box_left, box_right = x - w/2, x + w/2
    box_top, box_buttom = y - h/2, y + h/2
    if coord_norm:
        box_left *= image.shape[1]
        box_right *=  image.shape[1]
        box_top *= image.shape[0]
        box_buttom *= image.shape[0]
    return cv.rectangle(image, (round(box_left),round(box_top)), (round(box_right),round(box_buttom)), color, width)


def draw_boxes2d(image, label_name, color=(255,0,0), width=4, filter=None, coord_norm=False):
    labels = open(label_name, 'r').readlines()
    labels = [ label.rstrip('\n').split(' ') for label in labels]
    filtered_labels = labels if filter is None else [ label for label in labels if label[0] in filter ]
    
    for label in filtered_labels:
        x,y,w,h = label[1:5]
        image = draw_box2d(image, float(x),float(y),float(w),float(h), color=color, width=width, coord_norm=True)
    
    return image


def draw_box3d(image, x, y, z, h, w, l, ry, color=(0,255,0), width=2, cube=True, flatten=False, debug=False):
    import project_utils as P
    from math import sin,cos,pi

    delta_x1 = cos(ry)*l/2 + sin(ry)*w/2
    delta_x2 = cos(ry)*l/2 - sin(ry)*w/2
    delta_z1 = sin(ry)*l/2 - cos(ry)*w/2
    delta_z2 = sin(ry)*l/2 + cos(ry)*w/2
    pts = np.asarray([
        [x+delta_x1, y-h/2, z+delta_z1],
        [x+delta_x2, y-h/2, z+delta_z2],
        [x-delta_x1, y-h/2, z-delta_z1],
        [x-delta_x2, y-h/2, z-delta_z2],
        [x+delta_x1, y+h/2, z+delta_z1],
        [x+delta_x2, y+h/2, z+delta_z2],
        [x-delta_x1, y+h/2, z-delta_z1],
        [x-delta_x2, y+h/2, z-delta_z2]
    ])

    RT_Mat = np.asarray([
        [1,0,0,0  ],
        [0,1,0,0.9],
        [0,0,1,0.8],
        [0,0,0,1  ]
    ])
    pts_rect = np.dot(np.linalg.inv(RT_Mat), P.pts_to_hom(pts).T).T
    pts_img = np.dot(pts_rect, P.get_P_rect_Mat().T)
    pts_box = np.round((pts_img[:,0:2].T / pts_img[:,2]).T).astype(np.int)
    #print(pts_box)

    if cube:
        cv.line(image, tuple(pts_box[0]), tuple(pts_box[1]), color, width)
        cv.line(image, tuple(pts_box[1]), tuple(pts_box[2]), color, width)
        cv.line(image, tuple(pts_box[2]), tuple(pts_box[3]), color, width)
        cv.line(image, tuple(pts_box[3]), tuple(pts_box[0]), color, width)

        cv.line(image, tuple(pts_box[4]), tuple(pts_box[5]), color, width)
        cv.line(image, tuple(pts_box[5]), tuple(pts_box[6]), color, width)
        cv.line(image, tuple(pts_box[6]), tuple(pts_box[7]), color, width)
        cv.line(image, tuple(pts_box[7]), tuple(pts_box[4]), color, width)
        
        cv.line(image, tuple(pts_box[0]), tuple(pts_box[4]), color, width)
        cv.line(image, tuple(pts_box[1]), tuple(pts_box[5]), color, width)
        cv.line(image, tuple(pts_box[2]), tuple(pts_box[6]), color, width)
        cv.line(image, tuple(pts_box[3]), tuple(pts_box[7]), color, width)

    if flatten:
        x_max = np.max(pts_box[:,0])
        x_min = np.min(pts_box[:,0])
        y_max = np.max(pts_box[:,1])
        y_min = np.min(pts_box[:,1])
        cv.rectangle(image, (x_min,y_min), (x_max,y_max), color, width)

    if debug:
        return image, [pts, pts_box]
    else:
        return image


def draw_boxes3d(image, label_name, color=(0,255,0), width=2, filter=None, cube=True, flatten=False, debug=False):
    import bbox3d
    labels = open(label_name, 'r').readlines()
    try:
        bboxes = [ bbox3d.from_kitti_annotation(label) for label in labels ]
    except IndexError:
        bboxes = [ bbox3d.from_simplified_annotation(label) for label in labels ]
    if cube == False and flatten == False:
        cube = True
    box_list = []

    for bbox in bboxes:
        if 0 <= bbox.z <= 70.4 and -40 <= bbox.x <= 40:
            if debug:
                image, pts = draw_box3d(image=image,
                                        x=bbox.x,
                                        y=bbox.y,
                                        z=bbox.z,
                                        h=bbox.h,
                                        w=bbox.w,
                                        l=bbox.l,
                                        ry=bbox.ry,
                                        color=color,
                                        width=width,
                                        cube=cube,
                                        flatten=flatten,
                                        debug=debug)
                box_list.append(pts)
            else:
                image = draw_box3d(image=image,
                                   x=bbox.x,
                                   y=bbox.y,
                                   z=bbox.z,
                                   h=bbox.h,
                                   w=bbox.w,
                                   l=bbox.l,
                                   ry=bbox.ry,
                                   color=color,
                                   width=width,
                                   cube=cube,
                                   flatten=flatten,
                                   debug=debug)
    
    if debug:
        return image, box_list
    else:
        return image


def output_image(output_dir, image_name, image, box_list=None):
    print(os.path.join(output_dir, image_name))
    if box_list is not None:
        with open(os.path.join(output_dir, image_name[:-4]+'.txt'), 'w') as f:
            for box in box_list:
                print(box, file=f)
    return cv.imwrite(os.path.join(output_dir,image_name), image)


def main(image_dir='', label_dir='', output_dir='output', show_image = False, coord_norm=False, filter=None, mode='2d', cube=True, flatten=False, color=(0,0,255), debug=False):
    assert os.path.exists(image_dir) and os.path.exists(label_dir)
    assert mode == '2d' or mode == '3d'
    
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for image_name in os.listdir(image_dir):
        index = image_name[:-4]
        label_name = index+'.txt'
        if os.path.exists(os.path.join(label_dir, label_name)) == False:
            img = cv.imread(os.path.join(image_dir,image_name))
            output_image(output_dir, image_name, img)
        else:
            image = cv.imread(os.path.join(image_dir,image_name))
            if mode == '2d':
                img = draw_boxes2d(image, os.path.join(label_dir,label_name), filter=filter, coord_norm=coord_norm, color=color)
            else:
                box_list = None
                if coord_norm == False:
                    if debug:
                        img, box_list = draw_boxes3d(image, os.path.join(label_dir,label_name), filter=filter, cube=cube, flatten=flatten, color=color, debug=debug)
                    else:
                        img = draw_boxes3d(image, os.path.join(label_dir,label_name), filter=filter, cube=cube, flatten=flatten, color=color, debug=debug)
                else:
                    raise NotImplementedError
            output_image(output_dir, image_name, img, box_list)
        if show_image:
            cv.imshow(image_name, img)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--image_dir', default=r'D:\GitHub\carla-dataset\detbase_opposite\2\object\training\image_2')
    args.add_argument('--label_dir', default=r'D:\GitHub\carla-dataset\detbase_opposite\2\object\training\pred')
    args.add_argument('--output_dir', default='output')
    args.add_argument('--show_image', action='store_true', default=False)
    args.add_argument('--coord_norm', action='store_true', default=False)
    args.add_argument('--filter', default=None, nargs='*', type=str)
    args.add_argument('--color', default=(0,0,255), nargs=3, type=int)
    args.add_argument('--plot3d', action='store_true', default=False)
    args.add_argument('--cube', action='store_true', default=False)
    args.add_argument('--flatten', action='store_true', default=False)
    args.add_argument('--debug', action='store_true', default=False)
    args = args.parse_args()
    main(image_dir=args.image_dir,
         label_dir=args.label_dir,
         output_dir=args.output_dir,
         show_image=args.show_image,
         coord_norm=args.coord_norm,
         filter=args.filter,
         mode='3d' if args.plot3d else '2d',
         cube=args.cube,
         flatten=args.flatten,
         color=args.color,
         debug=args.debug)
