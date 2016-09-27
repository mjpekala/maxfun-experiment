% input image
%f = double(imread('cloth.tiff'))/255;
f = double(imread('Lena.jpg'))/255;
plot_help(f)
title('input image');

% choose parameter
M = size(f,1);
%b = M/8;
b = M/10;  % mjp: generates 121 \approx 128 feature dimensions (like SIFT)
sigma = b;

% construct Gaussian Gabor frame
tic;
G = Gabor_construct(M,b,sigma);
toc;

% compute coefficients
tic;
coeff = Gabor_transform(f,G);
toc;

% mjp - added some checks and a visualization.
assert(size(f,1) == size(coeff,1));
assert(size(f,2) == size(coeff,2));

c_min_max = [min(coeff(:))  max(coeff(:))];
for ii = [1 2 10 50 size(coeff,3)]
    figure('Position', [100 100 1200 400]); 
    ha = tight_subplot(1, 3, [.025 .025], [.01 .05], .01);

    axes(ha(1)); 
    imagesc(abs(coeff(:,:,ii)));  set(gca, 'XTick', [], 'YTick', []);
    colormap gray;  colorbar();
    title(sprintf('abs(); dim %d (of %d)', ii, size(coeff,3)));

    axes(ha(2)); 
    imagesc(real(coeff(:,:,ii))); set(gca, 'XTick', [], 'YTick', []);
    colormap gray;  colorbar();
    title(sprintf('real(), dim %d (of %d)', ii, size(coeff,3)));

    axes(ha(3)); 
    imagesc(imag(coeff(:,:,ii))); set(gca, 'XTick', [], 'YTick', []);
    colormap gray;  colorbar();
    title(sprintf('imag(),  dim %d (of %d)', ii, size(coeff,3)));
end
