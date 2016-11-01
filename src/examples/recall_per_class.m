function [recall, y_all] = recall_per_class(y_hat_all, y)
% RECALL_PER_CLASS  Reports per-class recall
%
%    y_hat_all   : an (m x n x t) matrix where:
%                       m is the number of examples in the data set, 
%                       n is the number of algorithms used, and
%                       t is the number of trials
%
%    y           : an (m x t) matrix of true class labels.
%                 

[m,n,t] = size(y_hat_all);
assert(size(y,1) == m);
assert(size(y,2) == t);


y_all = sort(unique(y(:)));

recall = zeros(numel(y_all), n, t);

% calculate per class metrics
for tt = 1:t
    for ii = 1:length(y_all)
        idx = find(y(:,t) == y_all(ii));
 
        for jj = 1:n
            n_correct = sum(y_hat_all(idx,jj,tt) == y(idx));
            recall(ii,jj,tt) = 100 * n_correct / length(idx);
        end
    end
end


% (optional) show results to stdout
for ii = 1:length(y_all)
    fprintf(' y = %3d |', y_all(ii));
    for jj = 1:n
        fprintf(' %6.2f (%6.2f)|', mean(recall(ii,jj,:)), std(recall(ii,jj,:)));
    end
    fprintf('\n');
end

