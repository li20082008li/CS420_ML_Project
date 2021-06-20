# CS420_ML_Project

**SJTU 2021Spring CS420 Machine Learning Project** 

### Group Members

- Yuxiang Lu
- Xuechen Li
- Fan Mu

### Introduction

#### Task: Medical Image Segmentation 

Segment neural structure from EM images

#### Dataset: ISBI Challenge 2012

### Models

- FCNs (FCN8s, FCN16s, FCN32s)
- U-Net
- UNet++
- CE-Net
- CPFNet

### Requirement

- ```Pytorch & torchvision```
- ```keras``` (just for data augmentation)
- ```matplotlib```
- ```tqdm```

### Usage

#### Test trained models:

```python run.py --model MODEL_NAME```

#### Train model from dataset:

1. Perform data augmentation:

   ```python augmentation.py```

   The augmented images and labels will be put in the ```data_aug/``` folder, and be spilt into a train set of 700 images and a validation set of 50 images.

2. Train the model:

   ```python run.py --model MODEL_NAME --train```

For ```MODEL_NAME```, you can choose from:

```FCN8s, FCN16s, FCN32, UNet, UNet++, CENet, CPFNet```

 



