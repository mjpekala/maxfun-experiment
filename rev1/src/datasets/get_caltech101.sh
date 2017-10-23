#!/bin/bash

CT101=101_ObjectCategories.tar.gz
wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/$CT101
tar xvfz $CT101
rm $CT101
