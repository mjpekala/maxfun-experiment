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


%% load data
load(sprintf('feats_%s.mat', pc.feature_type));  % creates "feats" variable
p % show parameters used to create data

% split into train/test
cvo = cvpartition(feats.y, 'HoldOut', 0.4);

fprintf('[%s]: Using %d train and %d test examples\n', ...
        mfilename, sum(cvo.training), sum(cvo.test));



%% Hyperparameter selection 

% "one window" MAXFUN approximation
acc_best = -Inf; maxfun_oo_best_idx = 0;

for ii = 1:size(feats.maxfun_oo,3)
    X_train = feats.maxfun_oo(:, cvo.training, ii);
    y_train = feats.y(cvo.training);
    
    model = fitcecoc(X_train', y_train, 'KFold', 3);
    y_hat = kfoldPredict(model);
    acc = sum(y_hat(:) == y_train(:)) / numel(y_train);
    
    if acc > acc_best
        fprintf('[%s]: maxfun_oo accuracy improved to %0.3f with index %d\n', mfilename, acc, ii);
        acc_best = acc;
        maxfun_oo_best_idx = ii;
    end
end

% mixed pooling needs to select alpha
alpha_all = linspace(0, 1, 11);
acc_best = -Inf; alpha_best = 0;

for alpha = alpha_all
    X_train = feats.avgpool(:, cvo.training)*(1-alpha) + ...
              feats.maxpool(:, cvo.training)*alpha;
    y_train = feats.y(cvo.training);
    
    model = fitcecoc(X_train', y_train, 'KFold', 3);
    y_hat = kfoldPredict(model);
    acc = sum(y_hat(:) == y_train(:)) / numel(y_train);
    
    if acc > acc_best
        fprintf('[%s]: mixed pooling accuracy improved to %0.3f with alpha=%0.1f\n', mfilename, acc, alpha);
        acc_best = acc;
        alpha_best = alpha;
    end
end

X_mixed = feats.avgpool*(1-alpha_best) + feats.maxpool*alpha_best;


%% Evaluate performance of baseline strategies

eval_svm(feats.avgpool, feats.y, cvo.training, 'average pooling');
eval_svm(feats.maxpool, feats.y, cvo.training, 'maximum pooling');
eval_svm(X_mixed, feats.y, cvo.training, sprintf('mixed pooling strategy (%0.2f)', alpha_best));
eval_svm(feats.probpool, feats.y, cvo.training, 'stochastic pooling');


%% Evaluate MAXFUN (and variants)

eval_svm(feats.maxfun, feats.y, cvo.training, 'MAXFUN pooling');
eval_svm(feats.maxfun_oo(:,:,maxfun_oo_best_idx), feats.y, cvo.training, 'MAXFUN one window size');
eval_svm(feats.maxfun_centered, feats.y, cvo.training, 'centered MAXFUN pooling');

