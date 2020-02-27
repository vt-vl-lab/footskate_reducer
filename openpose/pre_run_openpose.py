#!/usr/bin/python3

# Before running `run_openpose.py` in the `motion_reconstruct`, please run this
# Use OpenPose to extract 2D Human Pose

from glob import glob
import os

# NOTE: Please modify this accordingly (you need to use absolutie path here)
root = '/home/ylzou/research/WACV2020/motion_reconstruct/demo/example'
dump = '/home/ylzou/research/WACV2020/motion_reconstruct/openpose_output/example'

# NOTE: You might need to change the file extension
vid_paths = sorted(glob(os.path.join(root, "*.mp4")))

for i, vid_path in enumerate(vid_paths[::-1]):
    vid_name = os.path.basename(vid_path)[:-4]
    out_here = os.path.join(dump, vid_name)
    if not os.path.exists(out_here):
        os.makedirs(out_here)
    else:
        print('{} already exists, skip!'.format(vid_name))
        continue
    # NOTE: Please specify the path to your OpenPose, you should use the version with foot detector (my version is 1.2)
    os.chdir('/home/vllab1/tools/openpose')
    # NOTE: JPG output is for sanity check, you can choose not to visualize for saving disk storage.
    ## Best quality, but requires a large GPU memory
    # cmd = './build/examples/openpose/openpose.bin --video "'+vid_path+'" --write_json "'+out_here+'" --net_resolution "1312x736" --scale_number 4 --scale_gap 0.25 --write_images "'+out_here+'" --write_images_format jpg'
    ## Body25 model
    cmd = './build/examples/openpose/openpose.bin --video "'+vid_path+'" --write_json "'+out_here+'" --scale_number 4 --scale_gap 0.25 --write_images "'+out_here+'" --write_images_format jpg'
    os.system(cmd)
