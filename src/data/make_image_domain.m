function [X,Y] = make_image_domain(sz, a, domain)
% MAKE_IMAGE_DOMAIN  Creates a grid of spatial indices or wavenumbers.
%
%   [X,Y] = make_image_domain([m,n], a, 'time');  % spatial grid
%   [U,V] = make_image_domain([m,n], a, 'freq');  % wavenumber grid
%
%    where,
%      [m,n]  := the number of sampling points in the spatial domain
%                m := number of rows in the image (y indices)
%                n := number of columns in the image (x indices)
%      a      := the spacing between samples in the spatial domain
%      domain := {'space', 'freq'}
%
%   The wavenumber domain is based on an assumption that the corresponding 
%   spatial domain was mapped to a frequency basis via the DFT.
%
%   In both spatial and freqency domain 0,0 will be in the center of the grid.
%   This differs from fft2(), which puts the 0,0 frequency in the upper left.
%   Use fftshift() to align indices properly in this case.
%    
%  REFERENCES
%    o http://www.mathworks.com/matlabcentral/answers/24965-fft2-function-in-matlab

% July 2015, mjp

if nargin < 3, domain = 'space'; end
m = sz(1);  n = sz(2);

switch(lower(domain))
  case{'t', 'time', 'space'}
    xd = (0:n-1) - floor(n/2);     % n samples centered on 0
    yd = (0:m-1) - floor(m/2);     % m samples centered on 0
    [X,Y] = meshgrid(xd/a, yd/a);

  case{'f', 'freq'}
    dx = 1/a; dy = 1/a;            % spacing between each column & row
    xd = 0:dx:dx*(n-1);
    yd = 0:dy:dy*(m-1);

    xNyq = 1/(2*dx);               % Nyquist of data in x dimension
    yNyq = 1/(2*dy);               % "  " y dimension

    dkx = 1/(n*dx);                % wavenumber (spatial frequency) increment
    dky = 1/(m*dy);                %  "  "

    xk = -xNyq : dkx : xNyq - dkx;
    yk = -yNyq : dky : yNyq - dky;
    
    [X,Y] = meshgrid(xk, yk);

  otherwise
    error('unrecognized domain argument');
end



