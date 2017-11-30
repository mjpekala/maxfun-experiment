function out = max_pooling(X)
% MAX_POOLING
%
%    X      : A single image w/ dimensions (rows x cols x n_channels)
[r,c,n] = size(X);
out = max(reshape(X, r*c, n));
