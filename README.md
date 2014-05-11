CascadeSVMs
===========

### Introduction

CascadeSVMs is an open source C++ implementation of the CascadeSVMs algorithm (bounded with LIBLINEAR) for highly imbalanced large-scale data (i.e., negative samples >> positive samples) learning.


### Compiling

To compile the code, you need to have the OpenCV library (tested with OpenCV-2.4.6).


### Training and Testing

The interface is designed for TRECVID Surveillance Event Detection. It can be modified according to your tasks.  

./CascadeSVMs -c train.control -p train

./CascadeSVMs -c test.control -p test

The configuration and parameter settings are embeded in the training and testing control files.


### Bugs and Extensions

If you find any bug or develop some extensions, please feel free to drop me a line.


### Related Publication

Please cite our paper if you use the code:

X. Yang, C. Yi, L. Cao, and Y. Tian. MediaCCNY at TRECVID 2012: Surveillance Event Detection. NIST TRECVID Workshop, 2012. [[PDF](http://yangxd.org/publications/papers/TRECVID_2012_SED.pdf)]


### License Conditions

Copyright (C) 2013 Xiaodong Yang 

Distibution code version 1.0 - 05/09/2014. The code is for research purpose only. 