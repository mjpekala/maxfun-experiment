
## Quick start

1.  Download the CIFAR-10 data set.  You can use the script [here](./data/get_cifar.sh) or manually download and unpack the archive.

2.  Edit the [preprocessing](./preprocess_cifar10.m) script for your choice of windows and feature generation function.  In particular, you may want to change *window_size*, *stride*, *feature_type*, as well as the feature-specific parameters.  Then run this script to generate features for various pooling methods.   Note you need to run set_path.m the first time.

3. Edit (if needed) and run the the [classify](classify_cifar10.m) script to evaluate performance.

Note that the current setup is not as complete (e.g. in terms of parameter selection, cross-validation) as possible; the intent here is to facilitate rapid experimentation with various features and pooling strategies.
