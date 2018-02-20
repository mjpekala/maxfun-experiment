
# MAXFUN Pooling Experiments

## Quick Start
1.  Download the Caltech-101 data.  You can use the script [here](./src/data/get_caltech101.sh) or manually download and unpack the archive.

2.  Edit the [preprocessing](./src/process_images.m) script to select a windowing and feature type (e.g. gabor edge, dyadic edge).  The parameters of primary interest are towards the top of the script in the PARAMETERS section.

3.  Run the process_images.m script (note that you will have to [set Matlab's path](./src/set_path.m) before calling the script the first time).  The image processing script will generate a .mat file with the various pooled features for all images.  If using the caltech-lean data set (the default) the script will also cache the data set so that it need not be re-created from scratch each time (this cached version is also what one will process to produce the optional CNN features).

4. To evaluate classification performance, run the [classification](./src/classify_images.m) script.  This will "evaluate" the pooling strategies using a simple SVM classifier.

## Variations

### Using CNN-derived features

If, for a given experiment, you would prefer to use feature maps from a CNN, this requires an additional intermediate processing step.  After creating the Caltech lean data set in step 3 above (see also below for more details on this data set) you'll need to use the codes in [nets.py](./src/transforms/CNN/nets.py) to extract the feature maps of interest.  Then, re-run steps 3-4 above using "raw" feature "transformation" (ie. no feature extraction) on the .mat file created by nets.py.  

Note that, as of this writing, nets.py only extracts features for a single layer of the InceptionV3 network; however, it should be easy to modify to work with any network provided in tensorflow slim.

### Implementing different pooling functions

To implement a different pooling baseline, one need only:

1. Create a new matlab function that implements the API used by the existing pooling functions (e.g. see [avg_pooling.m](./src/avg_pooling.m)).
2. Add your new pooling output to the feats structure in [process_images.m](./src/process_images.m).
3. Call your pooling function in the main feature extraction loop.
4. Add your pooling output to [classify_images.m](./src/classify_images.m).


## Dataset: Caltech-101 "Lean"
The Caltech-101 "lean" data set is a subset of Caltech-101, restricted to classes having between 80 and 130 instances.  Our motivation is to have a data set where each class has sufficient representation while maintaining balance among classes ( by default, Caltech-101 has a few classes with very large membership and also a fair number of classes with few members).  

The classes satisfying our "sufficient representation yet balanced" criteria are:

```
>> process_images
[load_caltech101]: reading images...please wait...
[load_caltech101_lean]: keeping class 10 (brain) with 98 instances
[load_caltech101_lean]: keeping class 12 (buddha) with 85 instances
[load_caltech101_lean]: keeping class 13 (butterfly) with 91 instances
[load_caltech101_lean]: keeping class 20 (chandelier) with 107 instances
[load_caltech101_lean]: keeping class 36 (ewer) with 85 instances
[load_caltech101_lean]: keeping class 45 (grand_piano) with 99 instances
[load_caltech101_lean]: keeping class 46 (hawksbill) with 100 instances
[load_caltech101_lean]: keeping class 49 (helicopter) with 88 instances
[load_caltech101_lean]: keeping class 50 (ibis) with 80 instances
[load_caltech101_lean]: keeping class 53 (kangaroo) with 86 instances
[load_caltech101_lean]: keeping class 54 (ketch) with 114 instances
[load_caltech101_lean]: keeping class 56 (laptop) with 81 instances
[load_caltech101_lean]: keeping class 63 (menorah) with 87 instances
[load_caltech101_lean]: keeping class 76 (revolver) with 82 instances
[load_caltech101_lean]: keeping class 82 (scorpion) with 84 instances
[load_caltech101_lean]: keeping class 87 (starfish) with 86 instances
[load_caltech101_lean]: keeping class 91 (sunflower) with 85 instances
[load_caltech101_lean]: keeping class 93 (trilobite) with 86 instances
```



## Preprocessing

Prior to pooling, we perform a number of preprocessing steps:

1.  All images are padded to make them square; e.g. a 100x120 pixel image will be padded to 120x120 pixels.  The image is kept centered when padding, e.g. in the previous example 10 rows would be added to the top of the image and 10 to the bottom.  We then resize all images to 128x128 pixels.  See [this script](./data/resize_square.m) for the precise details.
2. A feature extraction algorithm is applied to the input image (e.g. gabor edge).  In general, this transforms an NxNx3 RGB image into a NxNxD feature tensor.
3. The feature tensors spatially decomposed into (possibly overlapping) windows that define the pooling regions (across all feature dimensions).  The total number of feature dimensions after pooling will be a function of the original image size, the window size, and the number of feature dimensions D.


## Example Results

Using the CNN-derived features (outputs from first convolutional layer) and the default parameters we choose, one should see results of the form:

```
>> classify_images

p = 

         dataset: 'caltech-101-lean-iv3-layer1'
    feature_type: 'raw'
     window_size: [21 21]
          stride: 21
     maxfun_supp: [2 6]

[classify_images]: Using 975 train and 649 test examples
[classify_images]: maxfun_oo accuracy improved to 0.585 with index 1
[classify_images]: maxfun_oo accuracy improved to 0.594 with index 3
[classify_images]: maxfun_oo accuracy improved to 0.599 with index 4
[classify_images]: maxfun_oo accuracy improved to 0.608 with index 5
[classify_images]: mixed pooling accuracy improved to 0.551 with alpha=0.0
[classify_images]: mixed pooling accuracy improved to 0.573 with alpha=0.1
[classify_images]: mixed pooling accuracy improved to 0.581 with alpha=0.2
[classify_images]: mixed pooling accuracy improved to 0.594 with alpha=0.3
[eval_svm]: "average pooling" accuracy: 0.5763
[eval_svm]: "maximum pooling" accuracy: 0.5932
[eval_svm]: "mixed pooling strategy (0.30)" accuracy: 0.6287
[eval_svm]: "stochastic pooling" accuracy: 0.6502
[eval_svm]: "MAXFUN pooling" accuracy: 0.6225
[eval_svm]: "MAXFUN one window size" accuracy: 0.6626
[eval_svm]: "centered MAXFUN pooling" accuracy: 0.5347
```


## References

* [bou10] Boureau, Ponce & LeCun "A theoretical analysis of feature pooling in visual recognition" ICML 2010.
* [zie13] Zieler & Fergus "Stochastic Pooling for Regularization of Deep Convolutional Neural Networks" ICLR 2013.
* [gra15] Graham "Fractional Max-Pooling," [arXiv 2015](https://arxiv.org/abs/1412.6071).
* [rip15] Rippel, Snoek & Adams "Spectral Representations for Convolutional Neural Networks" NIPS 2015.
 * [lee16] Lee, Gallagher & Tu "Generalized Pooling Functions in Convolutional Neural Networks: Mixed, Gated and Tree" Artificial Intelligence and Statistics, 2016.

