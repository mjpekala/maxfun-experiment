function alpha_best = select_pool_alpha(X_avg, X_max, y)
% SELECT_POOL_ALPHA   Selects alpha to use when interpolating
%                     between max and avg pooling.
%
%   X_avg : (d x n) matrix of n examples, each of which has 
%                   d dimensions.
%   X_max : (d x n) another matrix of features.
%
%       y : n-dimensional vector of class labels
%    
%  This is for implementing the mixed pooling function described by
%  equation 1 in [lee16].
%
%  REFERENCES:
%    [lee16] Lee, Gallagher, Tu "Generalizing Pooling Functions in
%            CNNs: Mixed, Gated and Tree".
%      

% mjp, november 2017
   
n_folds = 3;           % we are somewhat data poor, so keep k small
alpha_vals = 0:.1:1;
verbose = true;

cvo = cvpartition(y, 'KFold', n_folds);
acc = zeros(length(alpha_vals), n_folds);


for fold_id = 1:n_folds
    X_train_avg = X_avg(:, cvo.training(fold_id));
    X_train_max = X_max(:, cvo.training(fold_id));
    y_train = y(cvo.training(fold_id));
    
    X_test_avg = X_avg(:, cvo.test(fold_id));
    X_test_max = X_max(:, cvo.test(fold_id));
    y_test = y(cvo.test(fold_id));
   
    
    for ii = 1:length(alpha_vals)
        alpha = alpha_vals(ii);
        X_train = X_train_avg * (1-alpha) + X_train_max * alpha;
        X_test = X_test_avg * (1-alpha) + X_test_max * alpha;
        
        model = fitcecoc(X_train', y_train);
        y_hat = predict(model, X_test');
        acc(ii,fold_id) = sum(y_hat(:) == y_test(:)) / numel(y_test);
        
        if verbose
            fprintf('[%s]: fold %d, alpha=%0.2f acc=%2.3f\n', ...
                    mfilename, fold_id, alpha, acc(ii,fold_id));
        end
    end
end


acc_aggregate = mean(acc,2);
[~,best_idx] = max(acc_aggregate);

alpha_best = alpha_vals(best_idx);
    
