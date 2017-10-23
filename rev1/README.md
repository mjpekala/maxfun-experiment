# maxfun-experiment
An empirical study comparing MAXFUN (citation pending) to other feature pooling techniques.  This is joint work by Weilin Li, Yiran Li and Mike Pekala under the direction of Professor Wojtek Czaja of the [Norbert Wiener Center](http://www.norbertwiener.umd.edu/) at the University of Maryland.

**As of this writing this is a work in progress and all codes are highly experimental.  This repository is currently intended for internal use only.**

## Quick Start

1.  Download and install third-party dependencies (see setup.m for details).  Note that as of this writing (2016) it can be challenging (but possible) to get Matlab working with Xcode command line tools (without installing all of Xcode) under OSX 10.11.  Technically this is not a supported configuration by Mathworks but with sufficient hacking it can be made to work.
2.  Update matlab's search path by running setup.m from within Matlab.  Be sure you are picking up libsvm's version of svmtrain() (vs. Matlab's function of the same name).
3.  Run unit tests (run "runtests" from within Matlab after navigating into the "tests" directory).  If everything is working properly all tests should pass.
4.  Run demos/examples (see the "examples" subdirectory).  Of interest might be
    *  **demo_gabor.m**  A quick demonstration of how to use the Gabor transform codes.
    *  **demo_sift_and_pooling.m**  Provides examples of how to run SIFT on images and pool the resulting feature maps (in the context of a synthetic classification problem).

