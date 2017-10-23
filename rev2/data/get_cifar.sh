#!/bin/bash
#
# Downloads the (matlab version of) the CIFAR-10 and CIFAR-100 data sets

CIFAR_10=cifar-10-matlab.tar.gz
CIFAR_100=cifar-100-matlab.tar.gz

wget https://www.cs.toronto.edu/~kriz/${CIFAR_10}
tar xvfz ${CIFAR_10}

wget https://www.cs.toronto.edu/~kriz/${CIFAR_100}
tar xvfz ${CIFAR_100}

