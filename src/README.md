

## Quick Start

1.  Download third-party dependencies (see setup.m for details).
2.  Update matlab's search path (run setup.m from within Matlab).
3.  Run unit tests (run "runtests" from within Matlab after navigating into the "tests" directory).
4.  Run experiments (see the "experiments" subdirectory).  For now, the only experiment is "classification_experiment_1.m".  Note that this experiment doesn't require the SPAMS toolbox (only needed for sparse coding).  Also, you can get away with using Matlab's native SVM implementation if desired (so you can avoid installing libsvm but you will have to make a small change to eval_svm.m to switch SVM libraries).


