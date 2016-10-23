function recall_per_class(Y_hat_all, y, class_names)
% RECALL_PER_CLASS  Displays per-class recall
%
%    Y_hat_all   : an (m x n) matrix where m is the number of
%                  examples in the data set an n is the number of 
%                  algorithms that one used to evaluate each of 
%                  the m instances.
%
%    y           : an (m x 1) vector of true class labels.
%
%    class_names : a (p x 1) cell array of class names, where
%                  p is the total number of classes in the data set.
%                 

assert(length(y) == size(Y_hat_all,1));

y_all = sort(unique(y));

% create class names if needed
if nargin >= 3
    assert(length(class_names) == length(y_all));
else
    class_names = {};
    for ii = 1:length(y_all)
        class_names{ii} = sprintf('y=%2d', y_all(ii)); 
    end
end


% If Y_hat_all has more than 2 dimensions, reshape into an (m x n)
% matrix.  An assumption is that all dimensions beyond the first 
% correspond to independent experiments.
if ndims(Y_hat_all > 2)
    sz = size(Y_hat_all);
    Y_hat_all = reshape(Y_hat_all, sz(1), prod(sz(2:end)));
end


% per class metrics
for ii = 1:length(y_all)
    fprintf('  %14s  |', class_names{ii});
    idx = find(y == y_all(ii));
 
    for jj = 1:size(Y_hat_all,2)
        n_correct = sum(Y_hat_all(idx,jj) == y(idx));
        recall = 100 * n_correct / length(idx);
        fprintf(' %6.2f (%3d) |', recall, n_correct);
    end
    fprintf('\n');
end
