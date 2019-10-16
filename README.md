# AKE
The implementation of our ICDM 2019 paper "Guiding Entity Alignment via Adversarial Knowledge Embedding" [AKE](http://ddl.escience.cn/ff/endH).
## Requirements
python 3.5  
 torch == 0.31  
   numpy == 1.14
## How to use
### Dataset
The data folder includes our propocessed data for training and testing.   
 The orginal datasets can be founded from [here](https://github.com/nju-websoft/JAPE). 
### Training 
python AKE.py/AKE_FR.py/AKE_MR.py # with default hyper-parameters 
### Testing 
python test.py
## Citation 
If you find the code is useful for your research, please cite this paper:
```
@inproceedings{AKE
author={Xixun Lin and Yang Hong and Jia Wu and Chuan Zhou and Bin Wang},
title={Guiding Entity Alignment via Adversarial Knowledge Embedding},
journal={ACM Transactions on Graphics (Proc. SIGGRAPH)},
year={2019}
}
‘’‘
