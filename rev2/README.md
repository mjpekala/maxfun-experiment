
## Quick start

1.  Download the Caltech-101.  You can use the script [here](./data/get_caltech101.sh) or manually download and unpack the archive.

2.  Edit the [preprocessing](./process_images.m) script to select a windowing and feature type (e.g. gabor edge, dyadic edge).  The parameters of primary interest are towards the top of the script in the PARAMETERS section.

3.  Run the process_images.m script.  Note that you will have to [set Matlab's path](set_path.m) before calling the script the first time.  This will generate a .mat file with maximum, average and maxfun pooling for all images.

4. To evaluate classification performance, run the [classify](classify_images.m) script.  This will evaluate maxfun pooling as well as a number of pooling strategies of the form $\alpha * max + (1 - \alpha)*avg$.  



## Caltech-101 "Lean"
The Caltech-101 "lean" data set is a subset of Caltech-101, restricted to classes having between 80 and 130 instances. 
Our motivation is to have a data set where each class has sufficient representation while maintaining balance among classes.  
By default, Caltech-101 has a few classes with very large membership and also a fair number of classes with few members.  
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

1.  All images are padded to make them square; e.g. a 100x120 pixel image will be padded to 120x120 pixels.  The image is kept centered when padding, e.g. in the previous example 10 columns would be added to the left of the image and 10 to the right.  We then resize all images to 128x128 pixels.  See [this script](./data/resize_square.m) for the precise details.
2. 


## Performance

Here we show an example for one particular experiment setup:

```
         dataset: 'caltech-101-lean'
    feature_type: 'dyadic-edge'
     window_size: [28 28]
          stride: 28
     maxfun_supp: [2 6]
```

In this case, maxfun pooling does not provide a clear improvement.  Note that maxfun pooling overwhelmingly chooses a pooling support dimension of 2 (ie. a 2x2 pooling region) for this setup.   Furthermore, it never chooses a pooling support dimension greater than 3:

```
>> x = hist(w_maxfun(:), [2,3,4,5,6])
x =
3701042       40654           0           0           0
```

In this case, maxfun pooling does not provide a clear improvment (in aggregate); it is likely that with very small pooling support sizes maxfun will behave very much like max pooling ($\alpha=1$).  It is an open question as to how best maxfun (or its inputs) can be scaled to exhibit more dynamic behavior.

```
Elapsed time is 25.382307 seconds.
[classify_images]: maxfun classification accuracy is 0.770

[classify_images]: took 21.74 seconds to fit and predict for dyadic-edge:0.00
[classify_images]: classification accuracy is 0.777

[classify_images]: took 21.97 seconds to fit and predict for dyadic-edge:0.10
[classify_images]: classification accuracy is 0.788

[classify_images]: took 22.26 seconds to fit and predict for dyadic-edge:0.20
[classify_images]: classification accuracy is 0.779

[classify_images]: took 22.08 seconds to fit and predict for dyadic-edge:0.30
[classify_images]: classification accuracy is 0.781

[classify_images]: took 22.26 seconds to fit and predict for dyadic-edge:0.40
[classify_images]: classification accuracy is 0.785

[classify_images]: took 22.20 seconds to fit and predict for dyadic-edge:0.50
[classify_images]: classification accuracy is 0.773

[classify_images]: took 22.67 seconds to fit and predict for dyadic-edge:0.60
[classify_images]: classification accuracy is 0.766

[classify_images]: took 22.06 seconds to fit and predict for dyadic-edge:0.70
[classify_images]: classification accuracy is 0.764

[classify_images]: took 22.28 seconds to fit and predict for dyadic-edge:0.80
[classify_images]: classification accuracy is 0.769

[classify_images]: took 24.32 seconds to fit and predict for dyadic-edge:0.90
[classify_images]: classification accuracy is 0.770

[classify_images]: took 24.62 seconds to fit and predict for dyadic-edge:1.00
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
