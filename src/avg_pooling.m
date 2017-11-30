function out = avg_pooling(X)
% AVG_POOLING  
%
%    X      : A single image w/ dimensions (rows x cols x n_channels)
[r,c,n] = size(X);
out = mean(reshape(X, r*c, n));
