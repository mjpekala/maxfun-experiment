% DEMO_GABOR_2  A quick demo of Gabor features together with maxfun pooling.
%
%  The maxfun pooling (pooling inspired by the maximal function) hones
%  in on regions of signals where values of large magnitude are
%  concentrated.  It is this spatial localization property that
%  distinguishes maxfun pooling from, say, taking the whole region
%  average of the modulus of a feature dimension/image.
%
%  Taking the max of the modulus of a feature dimension is similar
%  except that its support is a single pixel.  
%
%  It is very unlikely that maxfun will use a pooling region whose
%  cardinality is much larger than the minimum.  Special cases might
%  include an annulus or other structures where there is a "ring" of
%  relatively large values whose dimension is greater than the minimum
%  cardinality of maxpool.


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
ylabel('pooled value');

figure; imagesc(f); colormap('gray'); title('raw image');

% view a few poolings
for ii = [1:5 10 20 30 50 100 120]
    figure;
    imagesc(abs(coeff(:,:,ii)));  colorbar; %colormap('gray');
    title(sprintf('feature dimension %d (of %d)', ii, size(coeff,3)));
    hold on;
    % Note: row/col specify the upper left corner
    row = nfo.row(ii);  % upper left corner
    col = nfo.col(ii);  %  "    "
    w = nfo.w(ii);
    line( [col  col+w], [row row], 'color', 'r');
    line( [col  col+w], [row+w row+w], 'color', 'r');
    line( [col  col], [row row+w], 'color', 'r');
    line( [col+w  col+w], [row row+w], 'color', 'r');
    hold off;
end
