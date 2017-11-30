function out = avg_pooling(X)
% AVG_POOLING   Implements spatial average pooling.
%
%    X  : A single image w/ dimensions (rows x cols x n_channels)
%   out : a n_channels dimensional vector of pooled values
%

% mjp, nov 2017

[r,c,n] = size(X);
out = mean(reshape(X, r*c, n));
