% input image
%f = double(imread('cloth.tiff'))/255;
f = double(imread('Lena.jpg'))/255;
plot_help(f)
title('input image');

% choose parameter
M = size(f,1);
b = M/8;
sigma = b;

% construct Gaussian Gabor frame
tic;
G = Gabor_construct(M,b,sigma);
toc;

% compute coefficients
tic;
coeff = Gabor_transform(f,G);
toc;

% mjp - added.
assert(size(f,1) == size(coeff,1));
assert(size(f,2) == size(coeff,2));
for ii = [1 2 10 50 size(coeff,3)]
    plot_help(coeff(:,:,ii));
    title(sprintf('Gabor feature dimension %d', ii));
end
