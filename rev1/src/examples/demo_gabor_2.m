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
%    - However, this raises an important point.  If we have some prior 
%      knowledge regarding the spatial extent of a "relevant" feature, 
%      it may make sense to incorporate this prior by setting the minimum
%      maxpool measure accordingly...


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

% maxfun pooling
%
% Note: since there is a modulus built into maxfun, this pooling is
% implicitly working on the magnitude of the Gabor features!
min_width = 11;
fprintf('[%s]: pooling...please wait a few moments...\n', mfilename);
tic
[maxfun_pool, nfo] = spatial_pool(coeff, 'maxfun', min_width);
toc

% other pooling (of the modulus)
coeff_2d = reshape(coeff, size(coeff,1)*size(coeff,2), size(coeff,3));
avg_pool = mean(abs(coeff_2d), 1);
max_pool = max(abs(coeff_2d), [], 1);


%% visualization

figure; imagesc(f); colormap('gray'); title('(resized) input image');

figure; 
plot(1:numel(maxfun_pool), maxfun_pool, 'o', ...
     1:numel(avg_pool), avg_pool, '-', ...
     1:numel(max_pool), max_pool, '-');
title('pooled modulus of Gabor features');
legend('maxfun', 'avg', 'max');
xlabel('feature dimension');
ylabel('pooled value');


figure; 
plot(1:length(nfo.w), nfo.w, 'o-');
title(sprintf('size of support selected by maxpool (minimum is %d)', min_width));
xlabel('feature dimension');
ylabel('pooling window width selected');


% view a few poolings
for ii = [1:5 10 20 30 50 100 120]
    Xi = abs(coeff(:,:,ii));
 
    [~,idx] = max(Xi(:));
    [r_max,c_max] = ind2sub(size(Xi), idx);
    
    figure;
    imagesc(Xi);  colorbar; %colormap('gray');
    title(sprintf('feature dimension %d (of %d)', ii, size(coeff,3)));
    hold on;
    % Visualize the maxpool region.
    % Note: row/col specify the upper left corner
    row = nfo.row(ii);  % upper left corner
    col = nfo.col(ii);  %  "    "
    w = nfo.w(ii);
    line( [col  col+w], [row row], 'color', 'r');
    line( [col  col+w], [row+w row+w], 'color', 'r');
    line( [col  col], [row row+w], 'color', 'r');
    line( [col+w  col+w], [row row+w], 'color', 'r');
    
    % also show the maximum value
    plot(c_max, r_max, 'rd');
    hold off;
end
