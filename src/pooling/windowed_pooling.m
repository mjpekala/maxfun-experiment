function Xp = windowed_pooling(X, poolDim, f_pool_wi)
%  WINDOWED_POOLING  Pools over disjoint spatial regions.
%

win_pool_2d = @(I) cell2mat(cellfun(f_pool_wi, pooling_regions(I, poolDim), ...
                                    'UniformOutput', 0));

if ndims(X) == 2
    Xp = win_pool_2d(X);
    
elseif ndims(X) == 3
    X1 = win_pool_2d(X(:,:,1));
    
    Xp = zeros(size(X1,1), size(X1,2), size(X,3));
    Xp(:,:,1) = X1;
    for ii = 2:size(X,3)
        Xp(:,:,ii) = win_pool_2d(X(:,:,ii));
    end
    
else
    error('sorry, input image must have either 2 or 3 dimensions');
end

