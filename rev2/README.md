
## Quick start

1.  Download the CIFAR-10 data set.  You can use the script [here](./data/get_cifar.sh) or manually download and unpack the archive.

2.  Edit the [preprocessing](./preprocess_cifar10.m) script for your choice of windows and feature generation function.  In particular, you may want to change *window_size*, *stride*, *feature_type*, as well as the feature-specific parameters.  Then run this script to generate features for various pooling methods.   Note you need to run set_path.m the first time.

3. Edit (if needed) and run the the [classify](classify_cifar10.m) script to evaluate performance.

Note that the current setup is not as complete (e.g. in terms of parameter selection, cross-validation) as possible; the intent here is to facilitate rapid experimentation with various features and pooling strategies.

## Caltech-101 "Lean"
The Caltech-101 "lean" data set is a subset of Caltech-101, restricted to classes with between 80 and 130 instances. 
Our motivation is to have a data set where each class has sufficient representation and there is a fair amount of balance among classes.
The classes satisfying this criteria are:

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
