# AKE
The implementation of our ICDM 2019 paper "Guiding Entity Alignment via Adversarial Knowledge Embedding" [AKE](https://ieeexplore.ieee.org/document/8970718).
## Requirements
python 3.5.3  
  torch == 0.3.1  
   numpy == 1.15  
     scipy == 1.1.0  
        scikit-learn == 0.20.0
## How to use
### Dataset
tar -zxvf data.tar.gz data  
 The data folder includes our propocessed data JA-EN for training and testing.   
 The orginal datasets can be founded from [here](https://github.com/nju-websoft/JAPE). 
### Training 
 zsh train.sh/train_variants.sh    # training AKE and variants with default hyper-parameters 
### Testing 
python test.py
## Citation 
If you find the code is useful for your research, please cite this paper:
```
@inproceedings{lin2019:AKE,
author={Lin, Xixun and Yang, Hong and Wu, Jia and Zhou, Chuan and Wang, Bin},
title={Guiding Entity Alignment via Adversarial Knowledge Embedding},
booktitle={IEEE International Conference On Data Mining (ICDM)},
year={2019}
}
```
