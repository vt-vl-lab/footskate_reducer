# Ground Contact Dataset

We have annotated several videos to create a new ground contact dataset, including training, validation, and test splits. You can download them using [Google Drive](https://drive.google.com/file/d/1sRFOHIea4lk79puN-epv28-Jq-h4i7wt/view?usp=sharing) or [VT server](https://filebox.ece.vt.edu/~ylzou/wacv2020reducing/WACV2020_data.tar).


## File Structure
```
contact_dataset: test split
  - labels: ground contact labels (fully annotated)
  - openpose: original OpenPose results
    - openpose_flow: parsed OpenPose results
  - videos: input video
  
Human3.6M: including training and validation videos
  - noisy_ground_2: ground contact labels (partially annotated, "-1" means not annotated)
  - openpose: parsed OpenPose results
 
MADS: including training videos
  - noisy_ground_2_mads: ground contact labels (partially annotated, "-1" means not annotated)
  - openpose: parsed OpenPose results
```

## Annotation
We use four numbers to represent the ground contact label for `Left Toe, Left Heel, Right Toe, Right Heel`. 0 means not in contact, 1 means in contact, -1 means not annotated.

## Credit
Some of the videos from [Human3.6M](http://vision.imar.ro/human3.6m/description.php) or [MADS](http://visal.cs.cityu.edu.hk/research/mads/#download).

**NOTE:** Due to data license of Human3.6M, we will not provide any visual data for the videos coming from this dataset. Instead, we only provide the pre-computed OpenPose resutls and ground contact labels.
