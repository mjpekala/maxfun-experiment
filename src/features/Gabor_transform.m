function coeff = Gabor_transform(f,G)
% GABOR_TRANSFORM computes the uniform covering frame coefficients 
%
% INPUT:
%  f                (real square matrix) image to transform
%  G                (tensor) of the Fourier transform of Gabor frame elements
%
% OUTPUT:
%  coeff            (tensor) frame coefficients fo f
%--------------------------------------------------------------------------
% Weilin Li ~ May 2016

% Fourier transform of f
F = fftshift(fft2(f)); 

% Fourier transform of frame coefficients
Coeff = G .* repmat(F,[1,1,size(G,3)]);

% take the inverse transform
coeff = ifft_help(Coeff);





