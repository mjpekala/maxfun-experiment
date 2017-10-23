function out = apply_transform(X, f)
% APPLY_TRANSFORM  Applies f along the feature dimension of X.
%
%    X      : A single image w/ dimensions (rows x cols x n_channels)
%

[rows,cols,feats] = size(X);

% process each input channel independently
out = {};
for ii = 1:size(X,3)
    out{ii} = f(X(:,:,ii));
end

% pack results from all channels into one big tensor
out = cat(3, out{:});
