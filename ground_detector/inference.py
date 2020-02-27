import argparse
import os
import shutil
import tempfile
from collections import OrderedDict

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import DatasetInference
from net import BasicTemporalModel
from utils import bin_to_bool, read_json

from dataset import REST_INDS


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Running inference for a given video.')

    # Data
    parser.add_argument('--vid_path', default='/home/ylzou/research/WACV2020/motion_reconstruct/demo/example/dance.mp4')
    parser.add_argument('--op_path', default='/home/ylzou/research/WACV2020/ground_detector/data/example/dance.npy')
    parser.add_argument('--time_window', type=int, default=3, help='How many nearby (past or future) frames to see, length=2*time_window+1')
    parser.add_argument('--pose_norm', type=int, default=1, choices=[0, 1], help='Normalize pose or not')
    parser.add_argument('--data_mode', type=str, default='kp', choices=['op', 'kp'], help='Choose which part of data to use')
    # Model
    parser.add_argument('--num_blocks', type=int, default=2, help='How many residual blocks to use')
    parser.add_argument('--num_features', type=int, default=512, help='Number of channels in the intermediate layers')
    parser.add_argument('--ckpt', default='pretrained/ckpt/model_best.pth', help='Path to load a pretrained model')
    # Optimization
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='How many workers for data loading')
    # Logging
    parser.add_argument('--out_dir', default='/home/ylzou/research/WACV2020/motion_reconstruct/ground_contact/example', help='Path to save models and logs')
    parser.add_argument('--out_prefix', default='')


    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print('Called with args:')
    print(args)

    ## Dataset
    dataset = DatasetInference(root=args.op_path, time_window=args.time_window, pose_norm=bin_to_bool(args.pose_norm), 
        output_mode=args.data_mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, 
        drop_last=False)
    dataiterator = iter(dataloader)
    steps_per_epoch = int(np.ceil(len(dataset)/args.bs))

    ## Model
    assert args.time_window > 0
    # Multiple frame
    if args.data_mode == 'op':
        in_channels = len(REST_INDS)*3
    elif args.data_mode == 'kp':
        in_channels = len(REST_INDS)*2
    else:
        raise NotImplementedError

    model = BasicTemporalModel(in_channels=in_channels, num_features=args.num_features, num_blocks=args.num_blocks, 
        time_window=args.time_window)
        
    ckpt = torch.load(args.ckpt)
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    # 0.5
    thres = 0.5
    preds = []
    for _ in range(steps_per_epoch):
        inputs = next(dataiterator)
        body = inputs
        body = body.cuda()

        with torch.no_grad():
            pred_prob = model(body)
        pred_prob = pred_prob.cpu().numpy()
        pred = pred_prob.copy()
        pred[pred_prob>=thres] = 1
        pred[pred_prob<thres] = 0
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)

    # Save
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    save_name = os.path.join(args.out_dir, args.out_prefix+args.op_path.split('/')[-1])
    save_name = save_name.replace('.npy', '_ground.npy')
    np.save(save_name, preds)

    # Visualize
    if args.vid_path:
        temp_dir = tempfile.mkdtemp(dir=args.out_dir)

        vid_cap = cv2.VideoCapture(args.vid_path)
        num_frame = len(dataset)
        kps = np.load(args.op_path)

        for t in range(num_frame):
            ret, im = vid_cap.read()

            #cv2.putText(im, 'Frame: {}/{}'.format(t, num_frame), (50, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
            kp = kps[t, :, :].copy()

            feet = kp[[19, 14, 22, 11], :2]
            for i in range(4):
                if preds[t, i] == 1:
                    color = (0, 255, 0)
                elif preds[t, i] == 0:
                    color = (0, 0, 255)
                else:
                    continue
                cv2.circle(im, (int(feet[i, 0]), int(feet[i, 1])), 5, color, -1)

            #import matplotlib.pyplot as plt
            # plt.imshow(im[:,:,::-1])
            # plt.show()
            out_name = os.path.join(temp_dir, '{:05d}.png'.format(t))
            cv2.imwrite(out_name, im)

        vid_save_name = args.out_prefix+args.op_path.split('/')[-1].replace('npy', 'mp4')
        cmd = '/usr/bin/ffmpeg -i "{}/%05d.png" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -r 50 -pix_fmt yuv420p -y "{}"'.format(temp_dir, os.path.join(args.out_dir, vid_save_name))
        os.system(cmd)
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    main()
