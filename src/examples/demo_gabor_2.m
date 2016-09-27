% DEMO_GABOR_2  A quick demo of Gabor features together with maxfun pooling.

%% Feature generation

% input image
f = double(imread('Lena.jpg'))/255;
f = imresize(f, [200, 200]);

% choose Gabor parameter
M = size(f,1);
b = M/10;  % mjp: generates 121 \approx 128 feature dimensions (like SIFT)
sigma = b;

% construct Gaussian Gabor frame and calc. Gabor coefficients.
tic;
G = Gabor_construct(M,b,sigma);
toc;
tic;
coeff = Gabor_transform(f,G);
toc;
size(coeff)


%% whole image pooling

% Note: since there is a modulus built into maxfun, this pooling is
% implicitly working on the magnitude of the Gabor features!
min_cardinality = 11;
fprintf('[%s]: pooling...please wait a few moments...\n', mfilename);
tic
[pooled, nfo] = spatial_pool(coeff, 'maxfun', min_cardinality);
toc


%% visualization

figure; 
stem(pooled);
title(sprintf('pooled gabor features (l2 norm=%0.2e)', norm(pooled,2)));
xlabel('feature dimension');
ylabel('pool value');

figure; imagesc(f); colormap('gray'); title('raw image');

% view a few poolings
for ii = [1:5 10 20 30]
    figure;
    imagesc(abs(coeff(:,:,ii)));  colormap('gray');
    title(sprintf('feature dimension %d (of %d)', ii, size(coeff,3)));
    hold on;
    scatter(nfo.col(ii), nfo.row(ii), 'ro');
    w2 = nfo.w(ii)/2;
    line( [nfo.col(ii)-w2  nfo.col(ii)+w2], [nfo.row(ii)+w2 nfo.row(ii)+w2], 'color', 'g');
    line( [nfo.col(ii)-w2  nfo.col(ii)+w2], [nfo.row(ii)-w2 nfo.row(ii)-w2], 'color', 'g');
    line( [nfo.col(ii)-w2  nfo.col(ii)-w2], [nfo.row(ii)-w2 nfo.row(ii)+w2], 'color', 'g');
    line( [nfo.col(ii)+w2  nfo.col(ii)+w2], [nfo.row(ii)-w2 nfo.row(ii)+w2], 'color', 'g');
    hold off;
end
