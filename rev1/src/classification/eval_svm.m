function [yHat, metrics] = eval_svm(Xtrain, ytrain, Xtest, ytest)
% EVAL_SVM   Trains and evaluates a linear SVM on a dataset.
%
%   [yhat, metrics] = eval_svm(Xtrain, ytrain, Xtest, ytest)
%
%    where,
%     Xtrain : an (nxd) matrix of n objects each with d features 
%              (i.e. objects-as-rows)
%     ytrain : an (nx1) vector of train class labels
%     Xtest  : an (mxd) matrix of m objects each with d features 
%              (i.e. objects-as-rows)
%     ytest  : an (mx1) vector of test class labels
%
%
%  This code uses a linear SVM for classification.  We could consider
%  other classifiers; however, our primary reference suggests that the
%  intersection SVM does not perform substantially differently from
%  linear SVM.
%
%  Formerly was using Matlab's built-in SVM, but for the 1-vs-all
%  problem, we instead want a multiclass capability (hence LIBSVM).

% mjp, april 2016


% (optional) data preprocessing
if 1
    % whiten data
    [Xtrain, mu, sigma] = zscore(Xtrain);
    Xtest = bsxfun(@minus, Xtest, mu);
    Xtest = bsxfun(@rdivide, Xtest, sigma);
end


if 0
    % Can use matlab's internal codes for binary problems
    svm = svmtrain(Xtrain, ytrain, 'kernel_function', 'linear');
    yHat = svmclassify(svm, Xtest);
    isCorrect = (yHat == ytest);
else
    % For multi-class problems, we'll use LIBSVM and a linear
    % model.  See LIBSVM README for parameter meanings.
    %
    % Note: under the hood, libsvm is implementing multi-class as a
    % collection of 1-vs-1 problems whose result is combined using
    % some voting strategy.  We can also cook up a true 1-vs-all
    % problem if desired.
    %
    % See Lazebnik "Beyond Bags of Features" paper for a standard
    % approach to setting up caltech 101 problem.
    %
    model = svmtrain(ytrain(:), Xtrain, '-t 0 -s 0 -q');
    [yHat, acc, prob] = svmpredict(ytest(:), Xtest, model);
    isCorrect = (yHat == ytest);
end

metrics.nCorrect = sum(isCorrect);
metrics.CM = confusionmat(ytest, yHat);           
metrics.acc = diag(metrics.CM) ./ sum(metrics.CM, 2);
