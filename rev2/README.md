
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

