function rv = mean_variance_ratio(V1, V2)
% MEAN_VARIANCE_RATIO  Computes abs(mu1 - mu2) / (sigma1 + sigma2)
%                      for two samples.
%
%  PARAMETERS
%   V1 : A n x p matrix of p samples taken from each of n random variables.
%   V2 : A n x q matrix of q samples taken from each of n random variables.
%
%  RETURNS
%    rv.mu1     (n x 1) vector of sample means from V1
%    rv.mu2     (n x 1) vector of sample means from V2
%    rv.sigma1  (n x 1) vector of sample stanard deviations (V1)
%    rv.sigma2  (n x 1) vector of sample stanard deviations (V2)
%    rv.phi     (n x 1) vector of absolute difference in means
%    rv.psi     (n x 1) vector of mean variance ratios
%
%  REFERENCES
%  [1] Boureau et al. "A Theoretical Analysis of Feature Pooling in
%      Visual Recognition," 2010.

% mjp, april 2016

assert(ndims(V1) >= 2);
assert(ndims(V2) >= 2);

rv.mu1 = mean(V1, 2);
rv.mu2 = mean(V2, 2);
rv.sigma1 = std(V1, 0, 2);
rv.sigma2 = std(V2, 0, 2);
rv.phi = abs(rv.mu1 - rv.mu2);
rv.psi = rv.phi ./ (rv.sigma1 + rv.sigma2);
