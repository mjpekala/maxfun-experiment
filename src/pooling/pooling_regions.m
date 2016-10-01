function R = pooling_regions(X, poolDim)
% POOLING_REGIONS Partitions an image X into disjoint pooling
%                 regions (aka regions of interest).
%
%    R = pooling_regions(X, poolDim)
%
%    where,
%       X       := an (m x n) image with 2 spatial dimensions
%                    - or -
%                  an (m x n x d) image with 2 spatial dimensions
%                                 and 1 feature dimension
%
%       poolDim := the side length of each square pooling region
%
%       R       := an (a x b) cell array where R(a,b) corresponds
%                  to the (a,b)th pooling region from X.
%
%    This function is essentially just a wrapper around mat2cell
%    that pads the input image so that it can be partitioned into
%    blocks of size poolDim^2.

% mjp, april 2016


% pad X so that its spatial dimension are multiples of poolDim
deltaH = mod(size(X,1), poolDim);
if deltaH > 0, 
    deltaH = poolDim - deltaH; 
end

deltaW = mod(size(X,2), poolDim);
if deltaW > 0, 
    deltaW = poolDim - deltaW; 
end;

X = [X ; zeros(deltaH, size(X,2), size(X,3))];
X = [X   zeros(size(X,1), deltaW, size(X,3))];


% Decompose X into disjoint regions
nPoolH = size(X,1) / poolDim;
nPoolW = size(X,2) / poolDim;

if size(X,3)  == 1
    R = mat2cell(X, poolDim * ones(nPoolH,1), poolDim * ones(nPoolW,1));
else
    R = mat2cell(X, poolDim * ones(nPoolH,1), poolDim * ones(nPoolW,1), size(X,3));
end

             
