function plot_help(A)
% PLOT_HELP plots the matrix A as a gray scale image
% INPUT:
%  A                (matrix) of values
%
% OUTPUT:
%                   figure of A as a gray scale image
%--------------------------------------------------------------------------
% Weilin Li ~ May 2016


figure
imagesc(real(A))
colormap gray
axis off