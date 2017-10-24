function [pool_value, pool_size, pool_loc] = maxfun_pooling(X, min_supp, max_supp, render)
% MAXFUN_POOLING  Pooling inspired by the discrete maximal function.
%
%    X      : A single image w/ dimensions (rows x cols x n_channels)

[rows,cols,n_channels] = size(X);

if nargin < 2, min_supp = 1; end
if nargin < 3, max_supp = min(rows,cols); end
if nargin < 4, render = false; end


pool_value = -Inf*ones(1,n_channels);   % the pooled value
pool_size = NaN*ones(1,n_channels);     % the size of the pooling region 
pool_loc = NaN*ones(1,n_channels);      % the location of the pooling region 


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


% (optional) visualize the pooling
if render
    for channel = 1:n_channels
        figure;
        imagesc(X(:,:,channel));
        colormap('bone'); colorbar;
    
        [r,c] = ind2sub([rows, cols], pool_loc(channel));
        w = pool_size(channel);
        r = floor(r - w/2);
        c = floor(c - w/2);
    
        line([c,c], [r, r+w], 'Color', 'r');
        line([c,c]+w, [r, r+w], 'Color', 'r');
        line([c,c+w], [r, r], 'Color', 'r');
        line([c,c+w], [r, r]+w, 'Color', 'r');
        
        title(sprintf('%0.2f (%d,%d ; %d)\n', pool_value(channel), r, c, w));
    end
end

