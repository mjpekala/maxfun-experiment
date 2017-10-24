function [pool_value, pool_size, pool_loc] = maxfun_pooling(X, max_supp)
% MAXFUN_POOLING  Pooling inspired by the discrete maximal function.
%
%    X      : A single image w/ dimensions (rows x cols x n_channels)

[r,c,n] = size(X);

if nargin < 2
    max_supp = min(r,c);
end


pool_value = -Inf*ones(1,n);   % the pooled value
pool_size = NaN*ones(1,n);     % the size of the pooling region used to compute value
pool_loc = NaN*ones(1,n);      % the location of the pooling region used to compute value


for channel = 1:n
    Xi = double(X(:,:,channel));
   
    for measure = 1:max_supp
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
