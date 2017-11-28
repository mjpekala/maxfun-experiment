function acc = eval_svm(X, y, desc)
% EVAL_SVM  Trains and tests an SVM on a data set.

rng(1066);  % for consistent results

model = fitcecoc(X, y, 'Kfold', 5);
y_hat = kfoldPredict(model);

acc = sum(y(:) == y_hat(:)) / numel(y);

if length(desc)
    fprintf('[%s]: "%s" accuracy: %0.3f\n', mfilename, desc, acc);
end

