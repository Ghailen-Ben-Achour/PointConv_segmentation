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

