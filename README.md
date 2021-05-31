# Spherical Multi-Modal Place Recognition for Heterogeneous Sensor Systems

![S2_Projections](https://user-images.githubusercontent.com/1336474/119963033-ee192100-bfa7-11eb-97e7-de71aa247fdc.png)

## Overview

## Installation


S2Loc was written using __PyTorch__ ([http://pytorch.org/](http://pytorch.org/)) and depends on a few libraries.
  * __s2cnn__: [https://github.com/jonas-koehler/s2cnn](https://github.com/jonas-koehler/s2cnn)
  * __lie_learn__: [https://github.com/AMLab-Amsterdam/lie_learn](https://github.com/AMLab-Amsterdam/lie_learn)
  * __pynvrtc__: [https://github.com/NVIDIA/pynvrtc](https://github.com/NVIDIA/pynvrtc)

Submodule references to these repositories can be found in the `lib` folder


## Usage

Clone this repository:
```
git clone git@github.com:ethz-asl/s2loc.git --recursive
```

To train a new model the use `train.py`.


## Reference

Our paper is available at 

*Bernreiter, Lukas, Lionel Ott, Juan Nieto, Roland Siegwart, and Cesar Cadena. 
"Spherical Multi-Modal Place Recognition for Heterogeneous Sensor Systems." 
2021 International Conference on Robotics and Automation (ICRA), vol. 2021-May, IEEE.* [[ArXiv](https://arxiv.org/abs/2104.10067)]

BibTex:
```
@INPROCEEDINGS{Bernreiter2021S2Loc,
 author={Bernreiter Lukas and Lionel Ott and Juan Nieto and Roland Siegwart and Cesar Cadena.},
 booktitle={2021 International Conference on Robotics and Automation (ICRA)},
 title={Spherical Multi-Modal Place Recognition for Heterogeneous Sensor Systems},
 year={2021},
 volume={},
 number={},
 pages={},
 doi={}}
```
