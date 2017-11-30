
## Quick start

1.  Download the Caltech-101 data.  You can use the script [here](./src/data/get_caltech101.sh) or manually download and unpack the archive.

2.  Edit the [preprocessing](./src/process_images.m) script to select a windowing and feature type (e.g. gabor edge, dyadic edge).  The parameters of primary interest are towards the top of the script in the PARAMETERS section.

3.  Run the process_images.m script.  Note that you will have to [set Matlab's path](./src/set_path.m) before calling the script the first time.  This will generate a .mat file with the various pooled features for all images.  If using caltech-lean (the default) it will also cache the data set so that it need not be re-created from scratch each time.

4. To evaluate classification performance, run the [classify](./src/classify_images.m) script.  This will "evaluate" the pooling strategies using a simple SVM classifier.


### Using CNN-derived features

If, for a given experiment, you would prefer to use feature maps from a CNN, this requires an additional intermediate processing step.  After creating the Caltech lean data set in step 3 above (see also below for more details on this data set) you'll need to use the codes in [nets.py](./src/transforms/CNN/nets.py) to extract the feature maps of interest.  Then, re-run steps 3-4 above using "raw" feature "transformation" (ie. no feature extraction) on the .mat file created by nets.py.  

Note that, as of this writing, nets.py only extracts features for a single layer of the InceptionV3 network; however, it should be easy to modify to work with any network provided in tensorflow slim.


## Caltech-101 "Lean"
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


## Performance

Here we show an example for one particular experiment setup:

```
         dataset: 'caltech-101-lean'
    feature_type: 'dyadic-edge'
     window_size: [28 28]
          stride: 28
     maxfun_supp: [2 6]
```

Note that, for this setup, maxfun pooling overwhelmingly chooses a pooling support dimension of 2 (ie. a 2x2 pooling region) for this setup.   Furthermore, it never chooses a pooling support dimension greater than 3:

```
>> x = hist(w_maxfun(:), [2,3,4,5,6])
x =
3701042       40654           0           0           0
```

In this case, maxfun pooling does not provide a clear improvment (in aggregate); it is likely that with very small pooling support sizes maxfun will behave very much like max pooling ($\alpha=1$).  It is an open question as to how best maxfun (or its inputs) can be scaled to exhibit more dynamic behavior.  We also observe there is fairly similar performance across all values of $\alpha$, which suggests that there is limited potential gain to be had by exploring the space between maximum and average pooling.  Note also this space is further reduced if we use windows of size 20x20.

```
Elapsed time is 25.382307 seconds.
[classify_images]: maxfun classification accuracy is 0.770

[classify_images]: took 21.74 seconds to fit and predict for alpha=0.00
[classify_images]: classification accuracy is 0.777

[classify_images]: took 21.97 seconds to fit and predict for alpha=0.10
[classify_images]: classification accuracy is 0.788

[classify_images]: took 22.26 seconds to fit and predict for alpha=0.20
[classify_images]: classification accuracy is 0.779

[classify_images]: took 22.08 seconds to fit and predict for alpha=0.30
[classify_images]: classification accuracy is 0.781

[classify_images]: took 22.26 seconds to fit and predict for alpha=0.40
[classify_images]: classification accuracy is 0.785

[classify_images]: took 22.20 seconds to fit and predict for alpha=0.50
[classify_images]: classification accuracy is 0.773

[classify_images]: took 22.67 seconds to fit and predict for alpha=0.60
[classify_images]: classification accuracy is 0.766

[classify_images]: took 22.06 seconds to fit and predict for alpha=0.70
[classify_images]: classification accuracy is 0.764

[classify_images]: took 22.28 seconds to fit and predict for alpha=0.80
[classify_images]: classification accuracy is 0.769

[classify_images]: took 24.32 seconds to fit and predict for alpha=0.90
[classify_images]: classification accuracy is 0.770

[classify_images]: took 24.62 seconds to fit and predict for alpha=1.00
[classify_images]: classification accuracy is 0.762

[classify_images]: there are 1624 examples total
[classify_images]: any alpha correct:        1421
[classify_images]: all alpha correct:        1016
[classify_images]: both avg and max correct: 1151
[classify_images]: only alpha in (0,1):      72
[classify_images]: only avg correct:         111
[classify_images]: only max correct:         87
[classify_images]: neither correct:          275
[classify_images]: 1118 of 1624 estimates do not change as a function of alpha
```

## References

* [bou10] Boureau, Ponce & LeCun "A theoretical analysis of feature pooling in visual recognition" ICML 2010.

* [zie13] Zieler & Fergus "Stochastic Pooling for Regularization of Deep Convolutional Neural Networks" ICLR 2013.

* [rip15] Rippel, Snoek & Adams "Spectral Representations for Convolutional Neural Networks" NIPS 2015.
 * [lee16] Lee, Gallagher & Tu "Generalized Pooling Functions in Convolutional Neural Networks: Mixed, Gated and Tree" Artificial Intelligence and Statistics, 2016.

