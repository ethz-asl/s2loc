# Spherical Multi-Modal Place Recognition for Heterogeneous Sensor Systems

![S2_Projections](https://user-images.githubusercontent.com/1336474/119963033-ee192100-bfa7-11eb-97e7-de71aa247fdc.png)

## Overview

In the context of robotics, place recognition is a fundamental problem for autonomous systems. It yields a estimated position of a robot in a prior map given the current observations.
We propose an end-to-end multi-modal approach that directly operates on camera images and LiDAR scans without the necessity of a feature extraction.
All modalities are projected onto a hypersphere and given as input to a spherical CNN that learns a unique embedding optimized for distinguishing between different places.

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

### Data Format
```
path_to_dataset/
    training_anchor_pointclouds/
    training_positive_pointclouds/
    training_negative_pointclouds/
    training_anchor_sph_images/
    training_positive_sph_images/
    training_negative_sph_images/
    anchor-poses.csv
    positive-poses.csv
    negative-poses.csv
    missions.csv
```
Images need to be projected separately, whereas pointclouds will be projected by the training set provider.
Missions are hash ids that are used to separate test and training places.
__An example training set will be provided soon.__

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
