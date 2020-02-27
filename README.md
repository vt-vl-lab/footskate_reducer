# Reducing Footskate in Human Motion Reconstruction with Ground Contact Constraints

Official implementation for [Reducing Footskate in Human Motion Reconstruction with Ground Contact Constraints](https://yuliang.vision/WACV2020/).

You would need different python environments to finish each step, please check the corresponding `README.md` file in each folder. You should also follow the instructions in those files to set up dataset path.

Please see the [project page](https://yuliang.vision/WACV2020/) for more details.


## NOTE
We use [OpenPose (version 1.2)](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to get 2D pose (Body_25 model), please follow their instructions to install it.


## Example
For video `demo/dance.mp4`, you can follow these steps to get our motion reconstruction result. But you might need to modify some lines to specify the path.

- [x] Check `openpose` folder: Use OpenPose to get per-frame detection json files. You need to modify some lines to specify the path. I have also put the raw OpenPose results in `demo/dance`.
- [x] Check `ground_detector` folder: We need to convert json files to npy format. See `op2npy.py`.
- [x] Check `ground_detector` folder: Get ground contact detection results. See `inference.py`.
- [ ] Check `motion_reconstruct` folder: We need to convert json files to h5 format. See `run_openpose.py`.
- [ ] Check `motion_reconstruct` folder: Last step, see `refine_video.py`

I have also put all the intermediate result files in the corresponding path, you should be able to run the last step directly (if you have specify the path correctly).


## TODO
- [ ] (Probably after ECCV submission) Code clean up for motion reconstruction part


## Citation
If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{zou2020reducing,
	    author    = {Zou, Yuliang and Yang, Jimei and Ceylan, Duygu and Zhang, Jianming and Perazzi, Federico and Huang, Jia-Bin}, 
	    title     = {Reducing Footskate in Human Motion Reconstruction with Ground Contact Constraints}, 
	    booktitle = {Winter Conference on Applications of Computer Vision},
	    year      = {2020}
	}
