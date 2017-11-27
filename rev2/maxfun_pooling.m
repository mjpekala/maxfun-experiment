function [pool_value, pool_size, pool_loc] = maxfun_pooling(X, min_supp, max_supp, render)
% MAXFUN_POOLING  Pooling inspired by the discrete maximal function.
%
%      X        : A single image w/ dimensions (rows x cols x n_channels)
%      min_supp : minimum pooling region dimension (side length)
%      max_supp : maximum pooling region dimension; use this to control computation time
%      render   : set to true to visualize the pooling (for debug only)
%
%  RETURNS:
%    pool_value : (1 x n_channels) vector of pooled values 
%    pool_size  : (1 x n_channels) vector of pooling region dimension
%    pool_loc   : (1 x n_channels) vector indicating index of pooling region
%
%  The latter two return values are just for debugging/visualization.


%% Parameters

if ndims(X) == 3
    [rows,cols,n_channels] = size(X);
else
    % deal with single channel images here
    [rows,cols] = size(X);
    n_channels = 1;
end


if nargin < 2, min_supp = 1; end
if nargin < 3, max_supp = min(rows,cols); end
if nargin < 4, render = false; end


%% Allocate space for the return values.

pool_value = -Inf*ones(1,n_channels);   % the pooled value
pool_size = NaN*ones(1,n_channels);     % the size of the pooling region 
pool_loc = NaN*ones(1,n_channels);      % the location of the pooling region 


%% the computation; implement as a set of convolutions.

for channel = 1:n_channels
    Xi = double(X(:,:,channel));
   
    for measure = min_supp:max_supp
        scale = measure * measure; % TODO: could try other scalings...
        filter = ones(measure, measure) / scale;
        
        result = conv2(Xi, filter, 'same');
        [value,idx] = max(result(:));
        
        if value > pool_value(channel)
            pool_value(channel) = value;
            pool_size(channel) = measure;
            pool_loc(channel) = idx;
        end
    end
end



%% (optional) visualize the pooling (or a subset thereof, for large # of features)

if render
    for channel = 1:min(n_channels,10)
        figure;
        imagesc(X(:,:,channel));
        colormap('bone'); colorbar;
    
        [r,c] = ind2sub([rows, cols], pool_loc(channel));
        w = pool_size(channel);
        r = floor(r - w/2) + .5;
        c = floor(c - w/2) + .5;
    
        line([c,c], [r, r+w], 'Color', 'r', 'LineWidth', 2);
        line([c,c]+w, [r, r+w], 'Color', 'r', 'LineWidth', 2);
        line([c,c+w], [r, r], 'Color', 'r', 'LineWidth', 2);
        line([c,c+w], [r, r]+w, 'Color', 'r', 'LineWidth', 2);

        title(sprintf('maxfun=%0.2f (row=%d, col=%d ; w=%d)\n', pool_value(channel), floor(r), floor(c), w));
    end
end

