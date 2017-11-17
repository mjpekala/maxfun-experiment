% PROCESS_IMAGES
%
%  Demonstrates (windowed) feature extraction and pooling.
%
%  It can be slow to extract features (depending on parameters and
%  pooling type).  Therefore, this script generates all the
%  features and pools them; other scripts are used for analysis 
%  (e.g. of classificatin performance).
% 
%  Before running this script the first time, call set_path.m
%  You also need to download the data set (see scripts in ./data).

% mjp, oct 2017

rng(1066);



%% PARAMETERS (change as desired for your experiment)

FIG_DIR = './Figures';

p.dataset = 'caltech-101-lean-iv3-layer1';
p.feature_type = 'raw';

if 0
    % These were for 128x128 images we process locally
    %p.window_size = [64,64];  p.stride = 32;   % 74-65
    %p.window_size = [32,32];   p.stride = 32;  % 77-75 (non-monotonic)
    p.window_size = [28,28];   p.stride = 28;  % 79-76
    %p.window_size = [20,20];   p.stride = 20;  %  79-78 nearly equal perf. across alpha
else
    p.window_size = [21,21];  p.stride = 21; % note: features from IV3 are 63x63
end


p.maxfun_supp = [2,6];


switch(lower(p.feature_type))
  case 'raw'
    % No local pre-processing.
    % This is not expected to work well if the input file is raw images.
    % However, we also use this when the input file contains some
    %   exogenously processed data (e.g. via a CNN)
    featurize = @(x) x;
    standardize = false;

  case 'gabor'
    f_xform = @(x) gabor_feature(double(x), 8, 8); % XXX: these may require tuning
    featurize = @(x) apply_transform(x, f_xform);
    standardize = false;
    
  case 'gabor-edge'
    % Note: Weilin recommends S=3
    f_xform = @(x) gabor_edge_feature(double(x), 3, 16); % XXX: these may require tuning
    featurize = @(x) apply_transform(x, f_xform);
    standardize = false;
    
  case 'dyadic-edge'
    % Note: Weilin recommends S=3
    f_xform = @(x) dyadic_edge_feature(double(x), 3, 16); % XXX: these may require tuning
    featurize = @(x) apply_transform(x, f_xform);
    standardize = false;
    
end



%% Load data set

switch (p.dataset)
  case 'cifar-10'
    batch_id = 1;
    data = load_cifar_10('./data/cifar-10-batches-mat', batch_id);

    % optional - reduce data set size
    fprintf('[%s]: WARNING - reducing data set size temporarily (for speed)\n', mfilename);
    data.X = data.X(:,:,:,1:1000);
    data.y = data.y(1:1000);
   
    
  case 'caltech-101-lean'
    % it is a little slow to load this data set so cache it after loading for first time
    cached_fn = 'caltech_101_lean.mat';
    
    if exist(cached_fn)
        load(cached_fn);
    else
        data = load_caltech101_lean('./data/101_ObjectCategories', 128);
        save(cached_fn, 'data', '-v7.3');
    end
   
  % UPDATE: added some CNN features
  case 'caltech-101-lean-iv3-layer1'
    load('caltech_101_lean_iv3_layer1.mat');
    data.X = permute(X_iv3, [2,3,4,1]);  % back to matlab-preferred order
    data.y = y(:)';  % force to be a row vector
    clear X_f y;
    
  otherwise
    error('unknown dataset!');
end


n_images = size(data.X,4);

if ~exist(FIG_DIR)
    mkdir(FIG_DIR);
end


%%
view_dataset(data.X, data.y, FIG_DIR);


if standardize
    % optional: normalize data
    % here we zero mean and unit variance along each pixel and channel, ie.
    %     mean(data.X(i,j,k,:)) = 0   (approximately)
    %     std(data.X(i,j,k,:)) = 1    (approximately)
    %
    fprintf('[%s]: WARNING - standardizing data\n', mfilename);
    mu = mean(data.X, 4);
    sigma = std(data.X, 0, 4);
   
    data.X = bsxfun(@minus, data.X, mu);
    data.X = bsxfun(@rdivide, data.X, sigma);
end



%% Preallocate space for features

% lazy way of determining how many feature dimensions we will have
% for each image.
x_dummy = featurize(data.X(:,:,:,1));

dummy = extract_all_windows(x_dummy, p.window_size, p.stride);
n_feats = size(dummy,3);
fprintf('[%s]: windowed data will have %d features\n', mfilename, n_feats);

feats.maxpool = zeros(n_feats, n_images);
feats.avgpool = zeros(n_feats, n_images);
feats.maxfun = zeros(n_feats, n_images);
feats.y = zeros(size(data.y));

% shuffle images (to remove correlation in labels)
feats.idx = randperm(n_images);


%% extract features
tic
last_chatter = -Inf;

w_maxfun = NaN*ones(n_feats,n_images);  % track maxfun support size (for debugging/analysis)


for ii = 1:n_images
    orig_idx = feats.idx(ii);  % index into original data set order
   
    % feature extraction
    x_i = data.X(:,:,:,orig_idx);                              % raw image
    x_f = featurize(x_i);                                      % filtered image 
    x_f = abs(x_f);                                            % NOTE: we always take modulus for now...
    x_fw = extract_all_windows(x_f, p.window_size, p.stride);  % filtered and windowed
   
    % pooling
    feats.maxpool(:,ii) = max_pooling(x_fw);
    feats.avgpool(:,ii) = avg_pooling(x_fw);
    feats.y(ii) = data.y(orig_idx);
    
    [feats.maxfun(:,ii), w_maxfun(:,ii), loc] = maxfun_pooling(x_fw, p.maxfun_supp(1), p.maxfun_supp(2));
    
    % sanity checks
    assert(all(feats.maxpool(:,ii) >= feats.avgpool(:,ii)));

    % status update
    runtime = toc;
    if runtime - last_chatter > 30
        fprintf('[%s] processed %d images (of %d) in %0.2f seconds (%s features)\n', ...
                mfilename, ii, n_images, runtime, p.feature_type);
        last_chatter = runtime;
    end
   
end

fprintf('[%s]: total runtime: %0.2f sec\n', mfilename, toc);
save(sprintf('feats_%s.mat', p.feature_type), 'feats', 'p', '-v7.3');


%% some visualization

figure; 
histogram(w_maxfun(:));
title('maxfun support sizes');
xlabel('support size');
ylabel('frequency');
saveas(gca, fullfile(FIG_DIR, 'maxfun_supp.png'));


figure('Position', [100, 100, 900, 300]);
subplot(1,3,1);
imagesc(x_i(:,:,1)); colorbar;  title('input');
subplot(1,3,2);
imagesc(x_f(:,:,1)); colorbar;  title(p.feature_type);
subplot(1,3,3);
imagesc(x_fw(:,:,1)); colorbar;  title('first window');
saveas(gca, fullfile(FIG_DIR, 'sample_windowing.png'));


figure;
for k = 1:10
    xk = x_fw(:,:,k);
    maxfun_pooling(xk, p.maxfun_supp(1), p.maxfun_supp(2), true);
    title(sprintf('%s : feature index %d', p.feature_type, k));
   
    fn = sprintf('sample_features_%d', k);
    saveas(gca, fullfile(FIG_DIR, fn));
end


figure;
subplot(1,2,1);
histogram(feats.maxpool);  title('Max Pooling features');
subplot(1,2,2);
histogram(feats.avgpool);  title('Avg. Pooling features');
