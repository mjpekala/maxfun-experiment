function acc = eval_svm(X, y, is_train, desc)
% EVAL_SVM  Trains and tests an SVM on a data set.

rng(1066);  % for consistent results across invocations

model = fitcecoc(X(:,is_train)', y(is_train));
y_hat = predict(model, X(:, ~is_train)');

y_test = y(~is_train);

acc = sum(y_test(:) == y_hat(:)) / numel(y_test);

if length(desc)
    fprintf('[%s]: "%s" accuracy: %0.4f\n', mfilename, desc, acc);
end

