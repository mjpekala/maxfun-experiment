function view_sift_features(X, row, col, Img)
% VIEW_FEATURES  Visualize SIFT feature data.
%
%    X   := a tensor of SIFT features with dimensions (height, width, 128)
%    row := which of the 4 SIFT rows to visualize
%    col := which of the 4 SIFT columns to visualize
%
% From the VLFeat documentation on SIFT descriptors:
% (http://www.vlfeat.org/api/sift.html#sift-intro-descriptor)
%
% """VLFeat SIFT descriptor uses the following convention. The y axis
% points downwards and angles are measured clockwise (to be consistent
% with the standard image convention). The 3-D histogram (consisting
% of 8×4×4=128 bins) is stacked as a single 128-dimensional vector,
% where the fastest varying dimension is the orientation and the
% slowest the y spatial coordinate. This is illustrated by the
% following figure."""

% mjp, april 2016

if nargin < 4, Img = []; end
if nargin < 3, col = 1; end
if nargin < 2, row = 1; end

assert( (1 <= row) & (row <= 4));
assert( (1 <= col) & (col <= 4));

xMin = min(X(:));
xMax = max(X(:));
showit = @(X) set(get(imagesc(X, [xMin, xMax]), 'Parent'), 'XTick', [], 'YTick', []);


offset = 8 * 4 * (row-1) + 8 * (col-1) + 1;


% Visualize spatial distribution of features across all 8 directional bins.
h = figure('Position', [100 100 1200 1200]);
ha = tight_subplot(3, 3, [.01 .01], [.01 .05], .01);
panels = [6 9 8 7 4 1 2 3];  % maps indices to correct angular bin in figure
for ii = 1:length(panels)
    axes(ha(panels(ii)));  showit(X(:,:,offset)); 
    offset = offset + 1;
    
    if panels(ii) == 2
        title(sprintf('SIFT features for bin (%d, %d)', row, col));
    end
end

if ~isempty(Img)
    axes(ha(5)); imagesc(Img);
    set(gca, 'XTick', [], 'YTick', []);
end

