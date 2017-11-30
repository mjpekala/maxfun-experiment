function f = ifft_help(F)
% IFFT_HELP computes the iFFT of the first 2 coordinates of the 3-d matrix F 
%
% INPUT:
%  F                (3-d-matrix) of the Fourier transform of several functions
%
% OUTPUT:
%  f                (3-d-matrix) of the inverse Fourier transform
%--------------------------------------------------------------------------
% Weilin Li ~ May 2016


f = zeros(size(F));

for j = 1:size(F,3)
    f(:,:,j) = ifft2(ifftshift(F(:,:,j)));
end
