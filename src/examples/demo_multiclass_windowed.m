%  DEMO_MULTICLASS_WINDOWED  
%
%   A variant of demo_multiclass_whole_image that partitions the
%   images into multiple "windows" prior to pooling.
%
%   This is a separate script 


% mjp, Oct. 2016


%% Experiment Parameters 

p_.seed = 9999;
%p_.nSplits = 10;  
p_.nSplits = 1; % TEMP  

%p_.classesToUse = 1:101;
p_.classesToUse = [1 2 3 4 6 13 20 24 48 56 95];
p_.experiment = sprintf('caltech-%d', numel(p_.classesToUse));


% --= Dataset Parameters =--
% NOTE: some of the paths below assume you are running this script from pwd.
%       Make sure to change as appropriate.
p_.imageDir = '../datasets/101_ObjectCategories';
p_.sz = [200 200];    % Gabor feature code requires square images
p_.window_dim = 50;
p_.nTrain = 30;       
p_.nTest = 30;       
p_.downsample = 1;    % feature space downsampling; 

% --= Gabor parameters =--
% choose b s.t., given p_.sz, Gabor has ~128 feature dimensions
p_.gabor.M = p_.sz(1);
p_.gabor.b = p_.gabor.M / 10;
p_.gabor.sigma = p_.gabor.b; 

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);

% --= SIFT parameters =--
p_.sift.size = 4;             % see dl_vsift


% --= output path stuff =--
p_.rootDir = fullfile('Results_dmw', [p_.experiment '_seed' num2str(p_.seed)]);

p_.fn.raw = 'raw_data.mat';
p_.fn.sift = 'feat_SIFT.mat';
p_.fn.gabor = 'feat_Gabor.mat';
p_.fn.results = 'results.mat';

p_  % show parameters to user


%% some helper functions

% shuffle: returns a randomly permuted copy of x
shuffle = @(x) x(randperm(numel(x)));

% make_dir: create a directory iff it does not already exist.
make_dir = @(dirName) ~exist(dirName) && mkdir(dirName);

% downsample: downsamples an image with dimensions
%   (rows x cols x n_channels x n_examples)
% 
% Note: making these intermediate lambda functions introduces some
% inefficiency.  However, for now we don't worry about this.
%
downsample_rows = @(I,p) I(1:p:end,:,:,:);
downsample_cols = @(I,p) I(:,1:p:end,:,:);
downsample = @(I,p) downsample_cols(downsample_rows(I,p),p);

% this function just runs sift (without macrofeatures)
run_sift = @(I) sift_macrofeatures(single(I), ...
                                   'step', p_.downsample, ...
                                   'sz', p_.sift.size, ...
                                   'subsamp', 0, ...
                                   'macrosl', 0);

run_gabor = @(I) downsample(Gabor_transform(I, G), p_.downsample);

feature_algos = {run_sift, run_gabor};

% pooling functions that do not require hyper-parameter selection can
% be defined here.
max_pooling = @(X) spatial_pool(X, 'max');
avg_pooling = @(X) spatial_pool(X, 'avg');
avg_abs_pooling = @(X) spatial_pool(abs(X), 'max');
ell2_pooling = @(X) spatial_pool(X, 'pnorm', 2);


%% Setup

rng(p_.seed, 'twister');
timestamp = datestr(now);
overallTimer = tic;

make_dir(p_.rootDir);
diary(fullfile(p_.rootDir, ['log_multiclass_' timestamp '.txt']));
loadedRawImagesYet = false;


%% load raw images
if ~exist('data', 'var')  % TEMP - remove me later!!
    data = load_image_dataset(p_.imageDir, p_.sz);
    if numel(data.y) == 0
        error('failed to load dataset!  Do the files exist?');
    end

    % pare down to classes of interest
    if ~isempty(p_.classesToUse)
        fprintf('[%s]: considering a %d class problem\n', mfilename, numel(p_.classesToUse));
        idx = ismember(data.y, p_.classesToUse);
        data.y = data.y(idx);
        data.X = data.X(:,:,idx);
    end
end



%% Run experiments

for splitId = 1:p_.nSplits
    experimentDir{splitId} = fullfile(p_.rootDir, sprintf('split_%0.2d', splitId));
    make_dir(experimentDir{splitId});
    
    if exist(fullfile(experimentDir{splitId}, p_.fn.results))
        continue;  % already completed this train/test split
    end

    %% determine the train/test split
    [isTrain, isTest] = select_n(data.y, p_.nTrain, p_.nTest);

    train.I = data.X(:,:,isTrain);
    train.y = data.y(isTrain);
    train.files = data.files(isTrain);
    train.idx = find(isTrain);
    train.foldId = assign_folds(train.y, 5);
 
    test.I = data.X(:,:,isTest);
    test.y = data.y(isTest);
    test.files = data.files(isTest);
    test.idx = find(isTest);

    
    %% run the experiment with each feature type
    Yhat = zeros(numel(test.y), 5, numel(feature_algos));  % stores classification results
    for algoId = 1:numel(feature_algos), 
        %------------------------------
        % generate features
        %------------------------------
        % compose the feature generation function with an image partitioning function
        f_algo = feature_algos{algoId};
        f = @(I) pooling_regions(f_algo(I), p_.window_dim, true);
       
        fprintf('[%s]: generating training data for algorithm %d (of %d)\n', mfilename, algoId, numel(feature_algos));
        feats.train.X = map_image(train.I, f);
        feats.train.y = train.y;
        
        fprintf('[%s]: generating test data for algorithm %d (of %d)\n', mfilename, algoId, numel(feature_algos));
        feats.test.X = map_image(test.I, f);
        feats.test.y = test.y;
 
        
        %------------------------------
        % select hyper-parameters
        %------------------------------
        % maxfun requires selecting a hyper-parameter
        if 0
            fprintf('[%s]: selecting hyper-parameter for MAXFUN pooling...\n', mfilename);
            tic
            maxfun_pooling_k = @(X, k) spatial_pool(X, 'fun', k);
            pMF = select_scalar_pooling_param(feats.train.X, feats.train.y, maxfun_pooling_k, 4:4:20, train.foldId); 
            maxfun_pooling = @(X) spatial_pool(X, 'fun', pMF);
            toc
        else
            maxfun_pooling = @(X) spatial_pool(X, 'fun', 15);
        end
       
        %------------------------------
        % evaluate each candidate pooling function
        %------------------------------
        poolfuncs = {max_pooling, avg_pooling, avg_abs_pooling, ell2_pooling, maxfun_pooling};
        for ii = 1:length(poolfuncs)
            fprintf('[%s]: evaluating pooling strategy %d (of %d)\n', mfilename, ii, length(poolfuncs));
            tic
            Xtrain = poolfuncs{ii}(feats.train.X);
            Xtest = poolfuncs{ii}(feats.test.X);
            % The transpose below is because the SVM codes want objects-as-rows.
            [yHat, metrics] = eval_svm(Xtrain', feats.train.y, Xtest', feats.test.y);
            Yhat(:, ii, algoId) = yHat;
            fprintf('[%s]: evaluation tool %0.2f (min)\n', mfilename, toc/60.);
        end
    end

    % save data to file
    fn = fullfile(experimentDir, p_.fn.results);
    save(fn, 'train', 'test', 'p_', 'Yhat', '-v7.3');

    clear feats;
end

return % TEMP


%% Post-processing / analysis
%
% This is simply aggregating results across the various train/test splits.

Yhat_sift = [];
Yhat_gabor = [];
for ii = 1:length(experimentDir), eDir = experimentDir{ii};
    % classification results for SIFT features
    fn = fullfile(eDir, 'svm_SIFT.mat');
    if exist(fn), 
        load(fn);
        Yhat_sift = [Yhat_sift ; Yhat];
    end
    
    % classification results for Gabor features
    fn = fullfile(eDir, 'svm_Gabor.mat');
    if exist(fn), 
        load(fn);
        Yhat_gabor = [Yhat_gabor ; Yhat];
    end
    
    % classification results for wavelet features
    fn = fullfile(eDir, 'svm_wavelet.mat');
    if exist(fn), 
        load(fn);
        Yhat_wavelet = [Yhat_wavelet ; Yhat];
    end
end


assert(all(Yhat_sift(:,1) == Yhat_gabor(:,1)));
classification_report(Yhat_sift(:,1), ...
                      [Yhat_sift(:,2:end) Yhat_gabor(:,2:end)], ...
                      {'SIFT+avg', 'SIFT+max', 'SIFT+pnorm', 'SIFT+fun', 'SIFT+sos', ...
                       'Gabor+avg', 'Gabor+max', 'Gabor+pnorm', 'Gabor+fun', 'Gabor+sos'});

diary off;