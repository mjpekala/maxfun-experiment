%  DEMO_MULTICLASS_WHOLE_IMAGE  
%
%   Compare the impact of different features and pooling strategies on
%   one-vs-all image classification problems.  Originally this code
%   started as a way to extend the experiments of [1].
%
%   This script is broken down into sections; earlier sections do data
%   preprocessing and later steps run the classification experiment.
%   Results are saved incrementally so that re-running the script at a
%   later time it will not have to recompute everything from scratch
%   (it takes awhile to generate some of the features).
%
%   There is also some attempt to avoid keeping too much data in
%   memory all at once (some feature sets can get large).  In
%   particular, we try to work with only one data split and one
%   representation at a time.  This introduces some "excess" file I/O;
%   improving upon the current implementation is certainly possible.
%
%
% REFERENCES:
%  [1] Boureau et al. "A Theoretical Analysis of Feature Pooling in 
%      Visual Recognition," 2010.
%  [2] Boureau et al. "Learning Mid-Level Features For Recognition," 2010.
%  [3] Chatfield et al. "The devil is in the details..." BMVC 2011.

% mjp, May 2016


%% Experiment Parameters 

p_.seed = 9999;
p_.experiment = 'caltech-101';
p_.nSplits = 10;  % FWIW, in section 2.3 in [1], the authors use 10 splits.

p_.classesToUse = [];


% NOTE: some of the paths below assume you are running this script from pwd.
%       Make sure to change as appropriate.
switch lower(p_.experiment)
  case {'caltech-101', 'caltech101'}
    p_.imageDir = '../datasets/101_ObjectCategories';
    %p_.sz = [200 300];    % assume fixed-size inputs in this script
    p_.sz = [200 200];    % XXX: keep square for Gabor feature code
    p_.nTrain = 30;       % see section 2.3 in [1] and also [3]
    p_.nTest = 30;        % see [3]
   
    % (optional): use only classes with >= 100 examples
    p_.classesToUse = [1 2 3 4 6 13 20 24 48 56 95];
      
  case {'kth_tips'}
    p_.imageDir = '/Users/pekalmj1/Data/KTH_TIPS';
    p_.sz = [200 200];
    p_.nTrain = 10;
    p_.nTest = 80 - p_.nTrain;
    
  otherwise
    error(sprintf('unrecognized experiment name: %s', p_.experiment));
end

p_.downsample = 4;            % feature space downsampling; 
                              % applies to all feature approaches

p_.gabor.M = p_.sz(1);        % see Gabor_construct.m
p_.gabor.b = p_.gabor.M / 8;  % see  " "
p_.gabor.sigma = p_.gabor.b;  % see  " "

p_.sift.size = 4;             % see dl_vsift

p_.wavelet.J = 4;             % see wavelet_feature.m

p_.rootDir = fullfile('DMC_Outputs', [p_.experiment '_seed' num2str(p_.seed)]);

rng(p_.seed, 'twister');
timestamp = datestr(now);
overallTimer = tic;

p_  % show parameters to user

% output file names used for intermediate results.
rawFn = 'raw_data.mat';
siftFn = 'feat_SIFT.mat';
gaborFn = 'feat_Gabor.mat';
waveletFn = 'feat_wavelet.mat';


%% some helper functions

% shuffle: returns a randomly permuted copy of x
shuffle = @(x) x(randperm(numel(x)));

% make_dir: create a directory only if it does not already exist.
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
run_sift = @(I) sift_macrofeatures(single(I), 'step', p_.downsample, 'sz', p_.sift.size, 'subsamp', 0, 'macrosl', 0);

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);
% mjp: !!! WARNING - trying complex->real !!!
run_gabor = @(I) single(abs(downsample(Gabor_transform(I, G), p_.downsample)));

run_wavelet = @(I) downsample(wavelet_feature(I, p_.wavelet.J), p_.downsample);

wi_sos_pool = @(X, k) spatial_pool(X, 'sos', k);
wi_mf_pool = @(X, k) spatial_pool(X, 'fun', k);


%% Pre-genenerate train/test splits for each experiment
%
% We do it this way in order to conserve memory.  Some datasets are
% sufficiently large that we can't featurize the entire data set in
% memory at once.

make_dir(p_.rootDir);
diary(fullfile(p_.rootDir, ['log_multiclass_' timestamp '.txt']));
loadedRawImagesYet = false;

for splitId = 1:p_.nSplits
    % create output file (if it does not already exist)
    experimentDir{splitId} = fullfile(p_.rootDir, sprintf('split_%0.2d', splitId));
    make_dir(experimentDir{splitId});
    
    fn = fullfile(experimentDir{splitId}, rawFn);
    if exist(fn, 'file')
        fprintf('[%s]: data already exists for train/test split %d; re-using...\n', ...
                mfilename, splitId);
        continue;
    else
        fprintf('[%s]: Creating train/test split %d (of %d)\n', mfilename, splitId, p_.nSplits);
    end
  
    % load raw images (if we haven't already)
    if ~loadedRawImagesYet
        data = load_image_dataset(p_.imageDir, p_.sz);
        if numel(data.y) == 0
            error('failed to load dataset!  Did you download it yet?');
        end

        loadedRawImagesYet=true;

        % pare down to classes of interest
        if ~ isempty(p_.classesToUse)
            fprintf('[%s]: reducing to a %d class problem!!\n', mfilename, numel(p_.classesToUse));
            idx = ismember(data.y, p_.classesToUse);
            data.y = data.y(idx);
            data.X = data.X(:,:,idx);
        end
    end
 
    [isTrain, isTest] = select_n(data.y, p_.nTrain, p_.nTest);

    train.I = data.X(:,:,isTrain);
    train.y = data.y(isTrain);
    train.files = data.files(isTrain);
    train.idx = find(isTrain);
 
    test.I = data.X(:,:,isTest);
    test.y = data.y(isTest);
    test.files = data.files(isTest);
    test.idx = find(isTest);

    % save data to file
    save(fn, 'train', 'test', 'p_', '-v7.3');
    clear train test isTrain isTest;
end

clear data;



%% Generate features

for ii = 1:length(experimentDir), eDir = experimentDir{ii};
    % input file
    fnRaw = fullfile(eDir, rawFn);
 
    % output files (one for each feature type)
    fnSIFT = fullfile(eDir, siftFn);
    fnGabor = fullfile(eDir, gaborFn);
    fnWavelet = fullfile(eDir, waveletFn);
 
    if exist(fnSIFT,'file') && exist(fnGabor,'file') && exist(fnWavelet,'file')
        continue  % all features have already been computed for this train/test split
    end
 
    % if we reached this point, some subset of the features needs to be
    % computed, so load the data.
    fprintf('[%s]: Loading data for train/test split %d (of %d)\n', mfilename, ii, p_.nSplits);
    load(fnRaw, 'train', 'test');

    
    %% SIFT
    if ~exist(fnSIFT, 'file')
        fprintf('[%s]: generating SIFT features...\n', mfilename);
        tic
        feats.train.X = map_image(train.I, run_sift);
        feats.train.y = train.y;
        feats.test.X  = map_image(test.I, run_sift);
        feats.test.y = test.y;
        toc
        
        fprintf('[%s]: training data set size: %s\n', mfilename, num2str(size(feats.train.X)));
        fprintf('[%s]: test data set size:     %s\n', mfilename, num2str(size(feats.test.X)));
        fprintf('[%s]: saving SIFT features...\n', mfilename);
        
        tic
        save(fnSIFT, 'feats', '-v7.3');
        clear feats; 
        toc
    end

    %% Gabor
    if ~exist(fnGabor,'file')
        fprintf('[%s]: generating Gabor features...\n', mfilename);
        tic
        % TODO: complex -> real?
        feats.train.X = map_image(train.I, run_gabor);
        feats.train.y = train.y;
        feats.test.X  = map_image(test.I, run_gabor);
        feats.test.y = test.y;
        toc
        
        fprintf('[%s]: training data set size: %s\n', mfilename, num2str(size(feats.train.X)));
        fprintf('[%s]: test data set size:     %s\n', mfilename, num2str(size(feats.test.X)));
        fprintf('[%s]: saving Gabor features...\n', mfilename);
        
        tic
        save(fnGabor, 'feats', '-v7.3');
        clear feats;
        toc
    end

    %% Wavelet
    if ~exist(fnWavelet,'file')
        fprintf('[%s]: generating wavelet features...\n', mfilename);
        tic
        feats.train.X = map_image(train.I, run_wavelet);
        feats.train.y = train.y;
        feats.test.X  = map_image(test.I, run_wavelet);
        feats.test.y = test.y;
        toc
       
        fprintf('[%s]: training data set size: %s\n', mfilename, num2str(size(feats.train.X)));
        fprintf('[%s]: test data set size:     %s\n', mfilename, num2str(size(feats.test.X)));
        fprintf('[%s]: saving wavelet features...\n', mfilename);
        
        tic
        save(fnWavelet, 'feats', '-v7.3');
        clear feats;
        toc
    end
end



%% Pooling and classification


% TODO: re-enable wavelet features once they are the same
% dimensions as the other feature maps.
%featFiles = {siftFn, gaborFn, waveletFn};
featFiles = {siftFn, gaborFn};

for ii = 1:length(experimentDir), eDir = experimentDir{ii};
    fprintf('[%s]: CLASSIFY - starting train/test split %d (of %d)\n', ...
            mfilename, ii, length(experimentDir));

    for jj = 1:length(featFiles), featFile=featFiles{jj};
        fnIn = fullfile(eDir, featFile);
        fnOut = strrep(fnIn, 'feat', 'svm');
        
        if exist(fnOut, 'file')
            fprintf('[%s]: estimates already exist: %s\n', mfilename, fnOut);
            continue;
        end
        
        fprintf('[%s]: analyzing features from file %s\n', mfilename, fnIn);
        load(fnIn, 'feats');
        
        % Step 1: select pooling hyper-parameters (for maxfun, sos)
        %   
        pcvTimer = tic;
        pFoldId = assign_folds(feats.train.y, 5);
        
        fprintf('[%s]: selecting hyperparameter for MAXFUN pooling...\n', mfilename);
        pMF = select_scalar_pooling_param(feats.train.X, feats.train.y, wi_mf_pool, 4:2:20, pFoldId); 
        
        fprintf('[%s]: selecting hyperparameter for SOS pooling...\n', mfilename);
        pSOS = select_scalar_pooling_param(feats.train.X, feats.train.y, wi_sos_pool, 6:2:24, pFoldId); 
        
        fprintf('[%s]: runtime for pooling parameter selection: %0.2f (min)\n', ...
                mfilename, toc(pcvTimer)/60.);

        % Step 2: pool features and classify.  
        %
        % For now, we are only doing "whole image" pooling; in the
        % future one could use pooling_regions() to implement spatial
        % pooling; of course, this raises the question as to how this
        % should be vectorized for the SVM.
        %
        % Thus, currently the input to the SVM is a nxm matrix where n
        % is the number of dimensions from the feature code (e.g. 128
        % for SIFT) and m is the number of examples.
        %
        poolfuncs = { @(X) spatial_pool(X, 'avg'), ...
                      @(X) spatial_pool(X, 'max'), ...
                      @(X) spatial_pool(X, 'pnorm', 2), ...
                      @(X) spatial_pool(X, 'fun', pMF), ...
                      @(X) spatial_pool(X, 'sos', pSOS)};

        Yhat = zeros(numel(feats.test.y), length(poolfuncs)+1);
        Yhat(:,1) = feats.test.y;  % first column is truth
        for kk = 1:length(poolfuncs), f_pool = poolfuncs{kk};
            Xtrain = f_pool(feats.train.X);
            Xtest = f_pool(feats.test.X);
            % The transpose below is because the SVM codes all want objects-as-rows.
            [yHat, metrics] = eval_svm(Xtrain', feats.train.y, Xtest', feats.test.y);
            Yhat(:,kk+1) = yHat(:);
        end

        % save truth and estimates to file for later analysis if desired
        save(fnOut, 'Yhat', '-v7.3');
        
        clear feats;
    end
end


%% Post-processing / analysis
%
% This is simply aggregating results across the various train/test splits.

Yhat_sift = [];
Yhat_gabor = [];
Yhat_wavelet = [];
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