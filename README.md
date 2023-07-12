# DLDS

## Overview:
This project is part of a component in DD2424 Deep Learning in Data Science 2023, at KTH university. The aim is to fine-tune different convolutional and transformer-based models to perform both binary and multi-class classification in a medical imaging setup. 



## Data: 
Two datasets were explored:

+ A [Brain Tumor Dataset](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) with 4479 MRI Images categorized into 44 classes [42 brain tumor classes and two normal classes].

+ [NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) consisting of over 112,000 frontal x-ray lung scans, with 14 disease classes and one class with healthy scans. This dataset is especially challenging due to it being a multi-category dataset (i.e. several labels for one image possible).



## Methods: 

Different Architectures Used: 
+ ResNet50
+ ResNetRS152
+ ViT
+ EfficientNetV2B1
+ ConvNeXtSmall
  
Fine-tuning & Training Techniques Considered (depending on the architecture): 
+ unfreezing different numbers of layers
+ hyperparameter search (different learning rates, learning rate schedulers, and number of epochs).
+ data augmentation
+ other custom layers after the base model
+ adding dropout
+ adding early stopping

Infrastructure: 
Training on Google Cloud (GPU access), several Tesla T4 GPUs on Vertex AI (notebooks and VM). 



## Results Summary: 
Fine-tuning ***ResNet50*** on the Brain Dataset in `BinaryClassification.py` using a batch size of 32, and 5 epochs, we reach a **test accuracy of 99%** for sinmple binary classification.

Multi-class Classification with ***ResNetRS152*** on the Brain Dataset in `ResNetRS152.py` (with data augmentation, and 20 epochs) reaches a final **test accuracy of 94%** for the multi-class setting.

Multi-class Classification with **EfficientNetV2B1** (334 layers) in `EfficientNetV2.py` with an augmentation layer and 80 training epochs, a final **test accuracy of 96%** was reached. 

Fine-tuning the **ConvNeXtSmall** (295 layers) with exponential decay learning scheduler, for 20 epochs, we get a final **test accuracy of 89%**. The same script `EfficientNetV2.py` is used here. 

Due to lack of training resources and time, fine-tuning the ViT, even with an added learning rate scheduler (exponential decay) and an augmentation layer, only a 77% final test accuracy was reached in the multi-class setting [see `vit_multiclass_augment`] and a 89% in the binary setting [see `vit_binary_2_decay`]. 

The general script for fine-tuning the Vision Transformer can be found in: `VIT.ipynb` while the remaining vit folders display result summary images, for different training regime variations. 



## References:
https://www.kaggle.com/datasets/nih-chest-xrays/data
https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c
https://www.kaggle.com/code/sanandachowdhury/transfer-learning-brain-tumor-classification
