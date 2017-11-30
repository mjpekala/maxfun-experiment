function G = Gabor_construct(M,b,sigma)
% GABOR_CONSTRUCT computes a Gaussian Gabor frame elements of size M x M.
%
% INPUT:
%  M                (int) number of pixels per dimension
%  b                (real number) frequency translation/spatial modulation
%  sigma            (real number) standard deviation of the Gaussian
%
% OUTPUT:
%  G                (tensor) first two indices are spatial coordinates
%                   and the third is the frame index. The frames from low
%                   to high frequencies
%--------------------------------------------------------------------------
% Weilin Li ~ May 2016

% number of frames
m = floor(M/(2*b));
numframes = (2*m+1)^2;

% make rectangular grid of with step size b,
% starting with the origin and increasing outwards
B = [0 0];
for r = 1:m
    E = [ b*r*ones(1,2*r+1); -b*r:b:b*r ]';
    N = [ b*(r-1):-b:b*(-r+1); b*r*ones(1,2*r-1)]';
    W = [ -b*r*ones(1,2*r+1); b*r:-b:-b*r ]';
    S = [ b*(-r+1):b:b*(r-1); -b*r*ones(1,2*r-1)]';
    B = [B ; E ; N ; W ; S];
end

% Fourier grid
[X,Y] = meshgrid(-M/2:M/2-1,-M/2:M/2-1);

% calculate the frames by translating a Gaussian
G = zeros(M,M,numframes);
for k = 1:numframes
    Z = -((X-B(k,1)).^2+(Y-B(k,2)).^2)/(2*sigma^2);
    G(:,:,k) = (sqrt(2*pi)*sigma)^(-1)*exp(Z);
end
