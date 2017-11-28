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
cvo = cvpartition(feats.y, 'HoldOut', 0.5);

fprintf('[%s]: Using %d train and %d test examples\n', ...
        mfilename, sum(cvo.training), sum(cvo.test));



%% Choose an alpha pooling value and evaluate performance
tic
alpha_pool = select_pool_alpha(feats.avgpool(:, cvo.training),...
                               feats.maxpool(:, cvo.training), ...
                               feats.y(cvo.training));
toc

X_test = feats.avgpool(:, cvo.test) * (1-alpha_pool) + feats.maxpool(:, cvo.test) * alpha_pool;
y_test = feats.y(cvo.test);

model = build_classifier(X_test', y_test);
y_hat = kfoldPredict(model);
acc = sum(feats.y(cvo.test)' == y_hat) / numel(y_hat);
fprintf('[%s]: mixed pooling classification (alpha=%0.2f) accuracy: %0.3f\n', mfilename, alpha_pool, acc);



%% Evaluate performance of max and average as well

model = build_classifier(feats.avgpool(:, cvo.test)', feats.y(cvo.test));
y_hat = kfoldPredict(model);
acc = sum(feats.y(cvo.test)' == y_hat) / numel(y_hat);
fprintf('[%s]: average pooling classification accuracy: %0.3f\n', mfilename, acc);

model = build_classifier(feats.maxpool(:, cvo.test)', feats.y(cvo.test));
y_hat = kfoldPredict(model);
acc = sum(feats.y(cvo.test)' == y_hat) / numel(y_hat);
fprintf('[%s]: maximum pooling classification accuracy: %0.3f\n', mfilename, acc);


%% Stochastic pooling of Zeiler and Fergus

model = build_classifier(feats.probpool(:, cvo.test)', feats.y(cvo.test));
y_hat = kfoldPredict(model);
acc = sum(feats.y(cvo.test)' == y_hat) / numel(y_hat);
fprintf('[%s]: stochastic pooling classification accuracy: %0.3f\n', mfilename, acc);


%% Evaluate maxfun

% TODO: hyperparameter selection for MAXFUN ???

model = build_classifier(feats.maxfun(:, cvo.test)', feats.y(cvo.test));
y_hat = kfoldPredict(model);
acc = sum(feats.y(cvo.test)' == y_hat) / length(y_hat);
fprintf('[%s]: maxfun classification accuracy is %0.3f\n', mfilename, acc);

