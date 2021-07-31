# Point Convolution

---

## Description
This project is based on PointConv paper. You can find the [arXiv](https://arxiv.org/abs/1811.07246) version here.
```bash
@article{wu2018pointconv,
  title={PointConv: Deep Convolutional Networks on 3D Point Clouds},
  author={Wu, Wenxuan and Qi, Zhongang and Fuxin, Li},
  journal={arXiv preprint arXiv:1811.07246},
  year={2018}
}
```
PointConv is a method that can efficiently perform convolution operations on non-uniformly sampled 3D point cloud data. This structure can achieve the same translation invariance as the 2D convolutional network, and the invariance of the permutation of the point order in the point cloud.

---

## How To Use

### Installation

This code is based on [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2) and [PointConv](https://github.com/DylanWusee/pointconv). Please install [Tensorflow](https://www.tensorflow.org/install) and follow the instructions in [PointNet++](https://github.com/charlesq34/pointnet2) to compile ```pointconv/tf_ops```
This code has been tested with:
- Python 2.7, Tensorflow 1.11.0, CUDA 9.0
- Python 3.7, Tensorflow 2.4.1, CUDA 11.0

#### Compile Customized TF Operators
First, please compile the TF operators under ```pointconv/tf_ops```. (Check ```tf_xxx_compile.sh``` under each ops subfolder) with this command.
```bash
sh tf_xxx_compile.sh
```
---

## Usage

### Virtual KITTI Semantic Segmentation

You can find the Dataset [here](https://github.com/VisualComputingInstitute/vkitti3D-dataset) or download it directly from this [link](https://www.vision.rwth-aachen.de/media/resource_files/vkitti3d_dataset_v1.0.zip).

#### Data Format

All files are provided as numpy ```.npy``` files.
Each file contains a ```N x F``` matrix, where ```N``` is the number of points in a scene and ```F``` is the number of features per point, in this case ```F=7```.
The features are ```XYZRGBL```, the 3D ```XYZ``` position, the ```RGB``` color and the ground truth semantic label ```L```. 



