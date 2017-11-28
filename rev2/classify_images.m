% CLASSIFY_IMAGES
%
%  --= For CIFAR-10 =--
%
%     Note that some of the weakest deep learning classifiers still get
%     ~75%  accuracy on CIFAR-10, e.g. see:
%         http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
%
%     I've seen some anectodal references to ~50% accuracy with a SVM; e.g. 
%        Martin & Vatsavai "Exploring Improvements for Simple Image Classification" 2013.
%        Le & Gopee "Classifying CIFAR-10 Images Using Unsupervised Feature & Ensemble Learning"
%
%     As of this writing, our "one layer" accuracy is relatively poor
%     (~40%); however, we have not exhausted all possible
%     pre-processing options.
%
%
%   --= For Caltech-101 Lean =--
%
%     This is a custom experiment, so there are not likely to be
%     existing baselines.  We will have to use our judgement in terms
%     of what is reasonable here (again, in a "one layer" context).
%

% mjp, oct 2017

rng(1066);


%% Experiment parameters
pc.feature_type = 'raw';
pc.alpha_all = linspace(0, 1, 11);
pc.classifier = 'svm';   % acc(SVM) > acc(KNN) for dyadic-edge(20,20,20)


switch(lower(pc.classifier))
  case 'svm'
    build_classifier = @(X,y) fitcecoc(X, y, 'Kfold', 5);
    
  case 'knn'
    build_classifier = @(X,y) fitcknn(X, y, 'NumNeighbors', 5, 'Standardize', 1, 'KFold', 5);
  otherwise
    error('unknown classifier');
end


%% load data
load(sprintf('feats_%s.mat', pc.feature_type));  % creates "feats" variable
p % show parameters used to create data

% split into train/test
cvo = cvpartition(feats.y, 'HoldOut', 0.6);

fprintf('[%s]: Using %d train and %d test examples\n', ...
        mfilename, sum(cvo.training), sum(cvo.test));



%% Evaluate performance of baseline strategies

% mixed pooling requires some hyperparameter selection
tic
alpha_pool = select_pool_alpha(feats.avgpool(:, cvo.training),...
                               feats.maxpool(:, cvo.training), ...
                               feats.y(cvo.training));

X_mixed = feats.avgpool(:, cvo.test) * (1-alpha_pool) + feats.maxpool(:, cvo.test) * alpha_pool;

eval_svm(X_mixed', feats.y(cvo.test), 'mixed pooling strategy');
eval_svm(feats.avgpool(:, cvo.test)', feats.y(cvo.test), 'average pooling');
eval_svm(feats.maxpool(:, cvo.test)', feats.y(cvo.test), 'maximum pooling');
eval_svm(feats.probpool(:, cvo.test)', feats.y(cvo.test), 'stochastic pooling');



%% Evaluate maxfun

% TODO: hyperparameter selection for maxfun??
eval_svm(feats.maxfun(:, cvo.test)', feats.y(cvo.test), 'MAXFUN pooling');

