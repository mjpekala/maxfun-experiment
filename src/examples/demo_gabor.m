% input image
%f = double(imread('cloth.tiff'))/255;
f = double(imread('Lena.jpg'))/255;
plot_help(f)

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

