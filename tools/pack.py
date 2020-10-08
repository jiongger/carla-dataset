import pickle
import os
import lib.bbox3d as bbox3d
import numpy as np
import matplotlib.pyplot as plt

def pack(label_dir='', target_file='pack.pkl', in_mode='p', out_mode='cube'):
    assert in_mode == 'p' or in_mode == 'g'
    assert out_mode == 'cube' or out_mode == 'flatten'
    assert os.path.exists(label_dir)
    label_names = os.listdir(label_dir)
    label_names = [ label_name for label_name in label_names 
                    if label_name[-4:]=='.txt' and label_name.startswith('00') ]
    label_names.sort()
    #print(label_names)

    if in_mode == 'p':
        Seq = []
        for label_name in label_names:
            CurrSeq = []
            labels = open(os.path.join(label_dir, label_name), 'r').readlines()
            for label in labels:
                bbox = bbox3d.from_kitti_annotation(label)
                if out_mode == 'cube':
                    CurrSeq.append([bbox.x, bbox.z, bbox.w, bbox.l, bbox.ry])
                else:
                    bbox.y, bbox.z = bbox.z, bbox.y
                    CurrSeq.append([list(vertex) for vertex in bbox.flatten(axis=2)])
            Seq.append(CurrSeq)    
    else:
        TrackId = []
        for label_name in label_names:
            labels = open(os.path.join(label_dir, label_name), 'r').readlines()
            for label in labels:
                bbox = bbox3d.from_simplified_annotation(label)
                if 0 <= bbox.z <= 70.4 and -40 <= bbox.x <= 40:
                    if label.rstrip('\n').split(' ')[1] not in TrackId:
                        TrackId.append(label.rstrip('\n').split(' ')[1])
        Seq_dict = {}
        for box_id in TrackId:
            Seq_dict[box_id] = []
        for label_name in label_names:
            labels = open(os.path.join(label_dir, label_name), 'r').readlines()
            for label in labels:
                bbox = bbox3d.from_simplified_annotation(label)
                box_id = label.rstrip('\n').split(' ')[1]
                if box_id in TrackId:
                    if out_mode == 'cube':
                        Seq_dict[box_id].append([bbox.x, bbox.z, bbox.w, bbox.l, bbox.ry])
                    else:
                        bbox.y, bbox.z = bbox.z, bbox.y
                        Seq_dict[box_id].append([list(vertex) for vertex in bbox.flatten(axis=2)])
        #print(TrackId)
        #print(Seq_dict[TrackId[0]], len(Seq_dict[TrackId[0]]))
        Seq = []
        for box_id in TrackId:
            Seq.append(Seq_dict[box_id])

    #print(Seq)
    #print(len(Seq))
    pickle.dump(Seq, open(target_file, 'wb'))

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--label_dir', default=r'D:\GitHub\carla-dataset\detbase_opposite\2\object\training\label_2')
    args.add_argument('--target', default='gt.pkl')
    args.add_argument('--input', default='g')
    args.add_argument('--output', default='cube')
    args.add_argument('--debug', action='store_true', default=False)
    args = args.parse_args()
    if not args.debug:
        pack(args.label_dir, args.target, args.input, args.output)
    else:
        pred = pickle.load(open('pred.pkl', 'rb'))
        gt = pickle.load(open('gt.pkl', 'rb'))
        fig,ax = plt.subplots()
        ax.set_xlim((-100,100))
        ax.set_ylim((-100,100))
        for track in gt:
            for patch in track:
                box = plt.Polygon([patch[0],patch[1],patch[2],patch[3]], fill=False, color='red', linewidth=0.5)
                ax.add_patch(box)
        for rec in pred:
            for patch in rec:
                box = plt.Polygon([patch[0],patch[1],patch[2],patch[3]], fill=False, color='blue', linewidth=0.5)
                ax.add_patch(box)
        plt.show()
