function out = extract_all_windows(X, sz, stride)
% EXTRACT_ALL_WINDOWS  
%
%    X      : A single image with dimensions (rows x cols x n_channels)
%             For graycale, n_channels should be 1.
%
%    sz     : window size (rows x cols)
%
%    stride : how far to step in rows and cols
%

assert(ndims(X) == 3);

out = {};

row_idx = 1:stride:(size(X,1) - sz(1) + 1);
col_idx = 1:stride:(size(X,2) - sz(2) + 1);

out = zeros(sz(1), sz(2), length(row_idx) * length(col_idx) * size(X,3));

idx = 1;
for channel = 1:size(X,3)
    for a = row_idx
        b = a + sz(1) - 1;
        
        for c = col_idx
            d = c + sz(2) - 1;
            out(:,:,idx) = X(a:b, c:d, channel);
            idx = idx + 1;
        end
    end
end
