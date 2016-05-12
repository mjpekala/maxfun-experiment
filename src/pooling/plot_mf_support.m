function plot_mf_support(X, row, col, w)
%
%   X          : an image that was maxpooled; (m x n) matrix
%   row, col   : the upper left corner of the support region
%   w          : the width/height of the support region

% mjp, may 2016

imagesc(X);  colormap(gray);

hold on
plot(col, row, 'ro');

line([col, col+w-1], [row, row], 'Color', 'r');
line([col, col+w-1], [row+w-1, row+w-1], 'Color', 'r');
line([col, col], [row, row+w-1], 'Color', 'r');
line([col+w-1, col+w-1], [row, row+w-1], 'Color', 'r');

hold off
