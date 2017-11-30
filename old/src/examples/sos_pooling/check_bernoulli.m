function rv = check_bernoulli(x, p, thresh)

if nargin < 3, thresh=.1; end

muHat = mean(x);
sigma2Hat = var(x);

mu = p;
sigma2 = p*(1-p);

% mean and variance should be with %5 of theoretical value
meanRelErr = (abs(mu-muHat) / mu);
varRelErr = (abs(sigma2 - sigma2Hat) / sigma2);

rv = (meanRelErr < thresh) & (varRelErr < thresh);
if ~rv, keyboard; end
