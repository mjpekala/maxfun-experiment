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

C = ones(size(Z_vals));
for ii = 1:size(Z_vals,1)-1
    largest_sum = max(Z_vals(ii:end,:), [], 1);
    C(ii,:) = Z_vals(ii,:) >= largest_sum;
end

pct_use_min_supp = sum(C,2) / size(C,2);

figure;
plot(pct_use_min_supp, 'o-');
grid on;
xlabel('min support dimension');
ylabel('freq. that min support is selected');
title('maxfun support analysis');


