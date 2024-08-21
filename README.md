# Enhancing Monocular 3-D Object Detection Through Data Augmentation Strategies
Code implementation of my paper [DA3D](https://ieeexplore.ieee.org/abstract/document/10497146). The code is based on [mmyolo](https://github.com/open-mmlab/mmyolo).
## Environment Installation

1. Create a new conda environment.
```shell
conda create -n DA3D python=3.7
```
2. Install the [pytorch](https://pytorch.org/get-started/previous-versions/).
```shell
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
3. Install custom libraries [cv_ops](https://github.com/jiayisong/cv_ops)
4. Install dependent libraries.
```shell
pip install -U openmim
mim install "mmengine==0.7.0"
mim install "mmcv==2.0.0rc4"
mim install "mmdet==3.0.0rc6"
mim install "mmdet3d==1.1.0rc3"
git clone https://github.com/jiasyiong/DA3D.git
cd DA3D
# Install albumentations
mim install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```
## Dataset Download
1. Image files
Download images from the [kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including 
*Download left color images of object data set (12 GB)*
and
*Download right color images, if you want to use stereo information (12 GB)*.
2. label files
The labeled files need to be converted, and for convenience I uploaded the converted files directly. They are [kitti_infos_test.pkl](https://drive.google.com/file/d/1WKZzsdcAjg9EVeLXLa5wAMbsZ4pxCRQU/view?usp=sharing), [kitti_infos_train.pkl](https://drive.google.com/file/d/1XzCNDc4CTuxtR2OfcdamnZTUaIWbtubs/view?usp=sharing), [kitti_infos_trainval.pkl](https://drive.google.com/file/d/1YkTG-_hG1T_eH5R43iVQrUYKw2-CU2Sc/view?usp=sharing), [kitti_infos_val.pkl](https://drive.google.com/file/d/1vbMq9bXo5w6B-ynoznIFGsU-vVhZUhRK/view?usp=sharing).
3. Unzip
Unzip the image file and organize it and the label file as follows.
```
kitti
├── testing
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
│   ├── image_3
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── training
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
│   ├── image_3
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── kitti_infos_test.pkl
├── kitti_infos_train.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_val.pkl
```
## Pre-training Model Download
未完待续

## Model Training
未完待续
## Model Testing
未完待续
## Citation

If you find this project useful in your research, please consider citing:

```latex
@ARTICLE{10497146,
  author={Jia, Yisong and Wang, Jue and Pan, Huihui and Sun, Weichao},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Enhancing Monocular 3-D Object Detection Through Data Augmentation Strategies}, 
  year={2024},
  volume={73},
  number={},
  pages={1-11},
  keywords={Three-dimensional displays;Object detection;Data augmentation;Task analysis;Pipelines;Cameras;Detectors;Autonomous driving;data augmentation;deep learning;monocular 3-D object detection},
  doi={10.1109/TIM.2024.3387500}
}
```

