function [pool_value, pool_size] = centered_maxfun_pooling(X)
% CENTERED_MAXFUN_POOLING  Pooling inspired by the discrete maximal function.
%
%    X   : A single image w/ dimensions (rows x cols x n_channels)
%
%  RETURNS:
%    pool_value : (1 x n_channels) vector of pooled values 
%    pool_size  : (1 x n_channels) vector of pooling region sizes
%                 This is just for debugging/analysis.

%-------------------------------------------------------------------------------
% Example:
%{
    fake_img = zeros(32,32,1);
    fake_img(10,10) = 100;
    [pv,ps] = centered_maxfun_pooling(fake_img);
    figure; imagesc(fake_img);
    line([17-ps/2, 17+ps/2], [17+ps/2 17+ps/2], 'Color', 'r');
    line([17-ps/2, 17+ps/2], [17-ps/2 17-ps/2], 'Color', 'r');
    line([17-ps/2, 17-ps/2], [17-ps/2 17+ps/2], 'Color', 'r');
    line([17+ps/2, 17+ps/2], [17-ps/2 17+ps/2], 'Color', 'r');
%}
%-------------------------------------------------------------------------------

% mjp, november 2017

[n_rows, n_cols, n_channels] = size(X);


% these could be made into function parameters later (if desired)
min_supp = 1;
max_supp = min(n_rows, n_cols);

epsilon = 1e-8;    % a small constant

% allocate storage for return values
pool_value = -Inf * ones(1,n_channels);
pool_size = zeros(1,n_channels);

% Determine center row and column.
rc = ceil(n_rows/2);  % rc := center row
cc = ceil(n_cols/2);  % cc := center column

% adjust center point based on parity of overall image size.
if mod(n_rows,2) == 0, rc = rc+1; end
if mod(n_cols,2) == 0, cc = cc+1; end


% main loop, over all possible measures.
for measure = min_supp:max_supp
    scale = measure * measure; 
    m_half = floor(measure/2);

    % determine which subset of the region to average.
    ra = rc - m_half;
    rb = rc + (measure - m_half - 1);

    ca = cc - m_half;
    cb = cc + (measure - m_half - 1);

    values_to_pool = X(ra:rb, ca:cb, :);
 
    % compute the average
    if measure > 1
      pooled_values_i = sum(sum(values_to_pool)) / scale;
    else
      pooled_values_i = values_to_pool;
    end
    pooled_values_i = squeeze(pooled_values_i)';
 
    % determine which values are larger and update outputs accordingly
    is_larger = pooled_values_i > (pool_value + epsilon);
    
    pool_value(is_larger) = pooled_values_i(is_larger);
    pool_size(is_larger) = measure;
end
