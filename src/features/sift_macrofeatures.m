function X = sift_macrofeatures(I, varargin)
% SIFT_FEATURES  A wrapper around vl_dsift() that implements the
% macrofeature capability described in [1].
%
%  PARAMETERS
%    I       : A grayscale image of type SINGLE.  
%    step    : SIFT sampling density
%    subsamp : spatial subsampling (partially determines maximum pool region size)
%    macrosl : macrofeature side length (see [1])
%    sz      : SIFT window size
%
%  REFERENCES
%    [1] Boureau et al. "Learning Mid-Level Features For Recognition," 2010.
%    [2] Boureau et al. "A Theoretical Analysis of Feature pooling," 2010.
%    [3] Lazebnik et al. "Beyond bags of features: ...," CVPR 2006.
%
%  NOTES
%    o I do not believe the input image I needs to be smoothed.  The
%      origins of this coding and pooling technique seem to be [3], which
%      explicitly states they do not smooth the input image (as I recall).

% mjp, april 2016

% these defaults are for the CALTECH-101 experiments described in
% [1,2] (which are not the same experiment, so keep this in mind).
parser = inputParser;
parser.addRequired('I', @(X) ndims(X) == 2);
parser.addParameter('step', 4, @(x) x > 0);     % see 3.2 in [1]
parser.addParameter('macrosl', 2, @(x) x > 0);  % see 3.2 in [1]
parser.addParameter('subsamp', 4, @(x) x > 0);  % see 3.2 in [1]
parser.addParameter('sz', 4, @(x) x > 0);       % see 2.3 in [2]
parser.parse(I, varargin{:});

sz = parser.Results.sz;
subsamp = parser.Results.subsamp;
macrosl = parser.Results.macrosl;
step = parser.Results.step;

% It seems the x, y indices in F use 1,1 as the upper left corner
% of the image (sift treats image as quadrant iv).
% This makes sense, as the data in d are then column-major.
[f,d] = vl_dsift(I, 'step', step, 'size', sz);

if 0
    figure;
    imagesc(I);  colormap(gray);
    hold on;
    plot(f(1,:), f(2,:), 'g.');
    hold off;
end

% Reshape into a tensor
nCol = length(unique(f(1,:)));
nRow = length(unique(f(2,:)));
nChan = size(d,1);  % usually 128
X = reshape(d.', nRow, nCol, nChan); 

if 0
    % sanity check
    v1 = squeeze(X(1,1,:));
    v2 = d(:,1);
    assert(all(v1 == v2));
end


% spatial subsampling
if subsamp > 1
    X = X(1:subsamp:end,:,:);
    X = X(:,1:subsamp:end,:);
end


% Implement macrosampling.
% This is just an exercise in careful reshaping and transposing.
if macrosl > 1
    % prune features so the dimensions are divisible by macrosl.
    r = floor(size(X,1) / macrosl);
    X = X(1:r*macrosl,:,:);
    c = floor(size(X,2) / macrosl);
    X = X(:,1:c*macrosl,:);
 
    X = permute(X, [3 1 2]);  % -> (chan, row, col)
    X = reshape(X, macrosl*size(X,1), size(X,2)/macrosl, size(X,3));
    X = permute(X, [1 3 2]);  % -> (chan, col, row)
    X = reshape(X, macrosl*size(X,1), size(X,2)/macrosl, size(X,3));
    X = permute(X, [3 2 1]);  % -> (row, col chan)
end
