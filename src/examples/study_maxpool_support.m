function Z_vals = study_maxpool_support(X)
%  STUDY_MAXPOOL_SUPPORT 
%
%   A function to help answer the question - "how often would
%   maxpool pick a pool size that isn't the smallest possible
%   cardinality?"
%
%     X := a tensor with dimensions (h x w x ...) where 
%             h is the image height, 
%             w the image width,
%             ... is at least one other dimension
%

% mjp, oct 2016

sz = size(X);
reshape_noncommutative = @(M) reshape(M, sz(1), sz(2), prod(sz(3:end)));

w_max = min(size(X,1), size(X,2));

% Turn X into a tensor with dimensions (m x n x d)
X = reshape_noncommutative(X);

% Process each example/feature separately
Z_vals = zeros(w_max, size(X,3));
for ii = 1:size(X,3)
    Xi = abs(X(:,:,ii));   % abs is to emulate maxfun
    Xi = Xi / sum(Xi(:));  % normalize
    Z = all_windowed_sums(Xi, 1:w_max);
    zi = max(max(Z, [], 1), [], 2);
    Z_vals(:,ii) = zi;
end

delta_w = diff(Z_vals);  % pos values = locations where support > min support
pct_monotonic = sum(delta_w(:) <= 0) / numel(delta_w);

figure;
boxplot(diff(Z_vals)')
title(sprintf('pct monotonic=%0.2f', pct_monotonic));
xlabel('pool window width');
ylabel('delta windowed sum (normalized)');

