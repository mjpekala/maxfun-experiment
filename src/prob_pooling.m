function out = prob_pooling(X)
% PROB_POOLING  The weighted-by-probability pooling of [zei13]
%
%    X      : A single image (rows x cols x n_channels)
%
%  See equations (4),(6) in [zei13].
%
%  REFERENCES
%    [zei13] Zeiler and Fergus "Stochastic Pooling for Regularization
%    of Deep Convolutional Neural Networks" 2013.


% mjp, november 2017

[r,c,n] = size(X);

out = zeros(n,1);

epsilon = 1e-12;   % to avoid divide-by-zero in calculations below

for ii = 1:n
    a_i = X(:,:,ii);                      % spatial pooling region
    p_i = a_i / (sum(a_i(:)) + epsilon);  % normalized activations; equation (4) in [zei13]
    out(ii) = sum(sum(p_i .* a_i));       % equation (6) in [zei13]
end
