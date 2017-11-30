function acc = eval_svm(X, y, is_train, desc)
% EVAL_SVM  Trains and tests an SVM on a data set.
%
%   X        : a (d x n) matrix of n examples, each with d features.
%   y        : an n-dimensional vector of class labels
%   is_train : an n-dimensional binary vector indicating which 
%              examples are for training.
%   desc     : a string description (only used for reporting)
%
%  RETURNS
%   acc      : classification accuracy
%
%  The main value of this function is that it provdes an easy
%  way to set the random number seed each time.  This helps
%  with consistency when comparing across feature types.

% mjp, november 2017

rng(1066);  % for consistent results across invocations

model = fitcecoc(X(:,is_train)', y(is_train));
y_hat = predict(model, X(:, ~is_train)');

y_test = y(~is_train);

acc = sum(y_test(:) == y_hat(:)) / numel(y_test);

if length(desc)
    fprintf('[%s]: "%s" accuracy: %0.4f\n', mfilename, desc, acc);
end

