CascadeSVMs
===========

### Introduction

CascadeSVMs is an open source C++ implementation of the CascadeSVMs algorithm (bounded with LIBLINEAR) for highly imbalanced large-scale data (i.e., negative samples >> positive samples) learning.


### Compiling

To compile the code, you need to have the OpenCV library (tested with OpenCV-2.4.6).


### Training and Testing

The current interface is designed for TRECVID Surveillance Event Detection.  

./CascadeSVMs -c train.control -p train
./CascadeSVMs -c test.control -p test

The configuration and parameter settings are embeded in training and testing control files.


### Related Publication

X. Yang, C. Yi, L. Cao, and Y. Tian. MediaCCNY at TRECVID 2012: Surveillance Event Detection. NIST TRECVID Workshop, 2012. [[PDF](http://yangxd.org/publications/papers/TRECVID_2012_SED.pdf)]


### Bugs and Extensions

If you find any bug or develop some extensions, please feel free to drop me a line.


### LICENSE CONDITIONS

Copyright (C) 2012 Xiaodong Yang 

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.