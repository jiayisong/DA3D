# Enhancing Monocular 3-D Object Detection Through Data Augmentation Strategies
Code implementation of my paper [DA3D](https://ieeexplore.ieee.org/abstract/document/10497146). The code is based on [mmyolo](https://github.com/open-mmlab/mmyolo).
## Environment Installation

### Create a new conda environment
```shell
conda create -n DA3D python=3.7
conda activate DA3D
```
### Install the [pytorch](https://pytorch.org/get-started/previous-versions/)
```shell
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```
### Install custom libraries [cv_ops](https://github.com/jiayisong/cv_ops)
### Install dependent libraries
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
### Image files
Download images from the [kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including 
*Download left color images of object data set (12 GB)*
and
*Download right color images, if you want to use stereo information (12 GB)*.
### label files
The labeled files need to be converted, and for convenience I uploaded the converted files directly. They are [kitti_infos_test.pkl](https://drive.google.com/file/d/17O_z-XXaxNZN-jxJn3OD9nkOZV29jtNg/view?usp=sharing), [kitti_infos_train.pkl](https://drive.google.com/file/d/1WKZzsdcAjg9EVeLXLa5wAMbsZ4pxCRQU/view?usp=sharing), [kitti_infos_trainval.pkl](https://drive.google.com/file/d/1YkTG-_hG1T_eH5R43iVQrUYKw2-CU2Sc/view?usp=sharing), and [kitti_infos_val.pkl](https://drive.google.com/file/d/1vbMq9bXo5w6B-ynoznIFGsU-vVhZUhRK/view?usp=sharing).
### Unzip
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
Modify the [configuration file](configs/rtmdet/det3d/rtmdet-3d_base.py#L22) appropriately based on the dataset location.
## Pre-training Model Download
Due to the presence of the PPP module, it is necessary to change the input channel of the convolution kernel in the first layer to 4. For the simplicity of the code, we directly give the modified pre-trained model weights. They are [cspnext-s](https://drive.google.com/file/d/1Rr3jS5US2k7eqyatphlTiU1pmVV1tB14/view?usp=sharing), [dla-34](https://drive.google.com/file/d/1lPiIZ2UtqyQURDSdyEmChTEIPeRFueOs/view?usp=sharing), and [v2-99](https://drive.google.com/file/d/1Xh5YKZQ81q9aU6hFP2ZC5TdWjd5eESzo/view?usp=sharing). Note that you have to specify the location of the pre-trained model weights in the configuration file. Or put it in the following location without modifying the configuration file.
```
DA3D
├── model_weight
│   ├── dla34-ba72cf86-base_layer_channel-4.pth
│   ├── cspnext-s_imagenet_600e_channel-4.pth
│   ├── depth_pretrained_v99_channel-4.pth
├── configs
├── ...
```
## Model Training
Similar to mmyolo, train with the following command. The batchsize used for the method in the paper is 8. When training with multiple gpu, pay attention to adjusting the size of batchsize in the configuration file.
```shell
# Single gpu
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/rtmdet/TabelV_line1.py
# Multi gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh configs/rtmdet/TabelV_line1.py 4
```


## Model Testing
Similar to mmyolo, test with the following command. 
```shell
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/rtmdet/TabelV_line1.py work_dirs/TabelV_line1/epoch_125.pth
```
When the test is complete, a number of txt files of the results are generated in *work_dir/result*. Then compressed into a zip it can be uploaded to the official [kitti server](https://www.cvlibs.net/datasets/kitti/user_submit.php).

## Running Result
The model I trained is given here. The following table is the same as Table V in the paper and the evaluation metrics are IOU=0.7, R40, AP_3D/AP_BEV on the validation set. 
| Network | Loss     | DA   | Easy           | Mod.           | Hard           |  Config  |  Download  |
|---------|----------|------|----------------|----------------|----------------|------|------|
| RTM     | SMOKE    |      | 8.57 / 11.65   | 7.89 / 10.94   | 7.00 / 9.88    | [config](configs/rtmdet/det3d/TableV_line1.py) | [model](https://drive.google.com/file/d/1zZXUzEj7t7tkEf1Lb8t4KcSqnGwvobNq/view?usp=sharing) \| [log](https://drive.google.com/file/d/1AMwjL0HOcf850uqApsQZfIBF50_fswf2/view?usp=sharing) |
| RTM     | SMOKE    | ✓    | 16.40 / 21.29  | 13.32 / 17.34  | 11.36 / 15.00  | [config](configs/rtmdet/det3d/TableV_line2.py) | [model](https://drive.google.com/file/d/1ZTdEGldUw06ocKgR_i2mfDwyFAf5dP0l/view?usp=sharing) \| [log](https://drive.google.com/file/d/1MDv6GF6eYEborwS0q3XCNK6lVRydG_1P/view?usp=sharing) |
| RTM     | MonoFlex |      | 14.38 / 18.90  | 11.27 / 15.07  | 9.65 / 12.98   | [config](configs/rtmdet/det3d/TableV_line3.py) | [model](https://drive.google.com/file/d/191CXdstSPyN_jgsRoZWJ2g8CiVEaR8Zx/view?usp=sharing) \| [log](https://drive.google.com/file/d/1U12dAQi9TZLOeYJBRrl5JcQD-5CAOqsK/view?usp=sharing) |
| RTM     | MonoFlex | ✓    | 21.79 / 25.95  | 17.04 / 20.86  | 14.87 / 18.23  | [config](configs/rtmdet/det3d/TableV_line4.py) | [model](https://drive.google.com/file/d/1yx_rO8G3g1yw5cVbMUMzqJ5q_vvqzFcm/view?usp=sharing) \| [log](https://drive.google.com/file/d/1XZt1HXGZempJCPxSuwpg2kAUr-o1tv0Z/view?usp=sharing) |
| DLA     | MonoFlex |      | 20.90 / 26.61  | 16.29 / 20.99  | 14.46 / 18.71  | [config](configs/rtmdet/det3d/TableV_line5.py) | [model](https://drive.google.com/file/d/17hFpGWr0fiGxO92xOm6N7pPNU9eXnVnz/view?usp=sharing) \| [log](https://drive.google.com/file/d/1p_zGkpfwAUVzSPnerZO8bz6T4bLWXJGP/view?usp=sharing) |
| DLA     | MonoFlex | ✓    | 25.66 / 31.56  | 21.68 / 26.73  | 19.27 / 23.80  | [config](configs/rtmdet/det3d/TableV_line6.py) | [model](https://drive.google.com/file/d/1B-nZsKB1jf7d04Y9Yy8r98omwfCuk24R/view?usp=sharing) \| [log](https://drive.google.com/file/d/1BIga8c3KaRbDioPjIraUXUQZP8jG2TZx/view?usp=sharing) |


The following table is the same as Table VI in the paper and the evaluation metrics are IOU=0.7, R40, AP_3D/AP_BEV on the test set through the official server. 
| Method          | Easy           | Mod.           | Hard            | Time | GPU    |  Config  |  Download  |
|-----------------|-----------------|-----------------|-----------------|------|--------|-------|-------|
| DA3D     | 27.76/36.83     | 20.47/26.92     | 17.89/23.41     | 22   | 2080Ti | [config](configs/rtmdet/det3d/TableVI_line1.py) | [model]() \| [log]() |
| DA3D*   | 30.83/39.50     | 22.08/28.71     | 19.20/25.20     | 22   | 2080Ti | [config](configs/rtmdet/det3d/TableVI_line2.py) | [model]() \| [log]() |
| DA3D** | 34.72/44.27     | 26.80/34.88     | 23.05/30.29     | 120  | 2080Ti | [config](configs/rtmdet/det3d/TableVI_line3.py) | [model]() \| [log]() |
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

