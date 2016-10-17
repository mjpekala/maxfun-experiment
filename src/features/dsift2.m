function X = dsift2(I, varargin)
% DSIFT2 A wrapper around vl_dsift that reshapes output into a tensor.
%
%  PARAMETERS
%    I       : A grayscale image (m x n matrix) of type SINGLE.  
%    ...     : Any parameters other parameters for vl_dsift()
%
%  RETURNS
%    X       : A tensor with dimensions m' x n' x d
%              where m' <= m, n' <= n and d is
%              the number of SIFT features (128 by default)
%

% mjp, october 2016


% It seems the x, y indices in F use 1,1 as the upper left corner
% of the image (sift treats image as quadrant iv).
% This makes sense, as the data in d are then column-major.
[f,d] = vl_dsift(I, varargin{:});

% Reshape into a tensor
nCol = length(unique(f(1,:)));
nRow = length(unique(f(2,:)));
nChan = size(d,1);  % usually 128
%if nChan ~= 128, warning('unexpected # feature dimensions'); end
X = reshape(d.', nRow, nCol, nChan); 
