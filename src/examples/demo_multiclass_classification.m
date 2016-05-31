%  DEMO_MULTICLASS_CLASSIFICATION  
%
%   Compare the impact of different features and pooling strategies on
%   one-vs-all image classification problems.  Originally this code
%   started as a way to repeat/extend the experiments of [1].
%
%   This script is broken down into sections; earlier sections do data
%   preprocessing and later steps run the classification experiment.
%   We save results incrementally as we go, so if you re-run the
%   script at a later time it will load the intermediate results from
%   file (vs recompute them from scratch).
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
%p_.nSplits = 10;  % FWIW, in section 2.3 in [1], the authors use 10 splits.
p_.nSplits = 2;  % TEMP - only using two splits while debugging
p_.nAtoms = 128;  % set to a non-positive value if you don't want
                  % sparse coded features.

switch lower(p_.experiment)
  case {'caltech-101', 'caltech101'}
    p_.imageDir = '/Users/pekalmj1/Data/caltech_101/101_ObjectCategories';
    %p_.sz = [200 300];    % assume fixed-size inputs in this script
    p_.sz = [200 200];    % TEMP - keep square for Gabor
    p_.nTrain = 30;       % see section 2.3 in [1] and also [3]
    p_.nTest = 30;        % see [3]
      
  case {'kth_tips'}
    p_.imageDir = '/Users/pekalmj1/Data/KTH_TIPS';
    p_.sz = [200 200];
    p_.nTrain = 10;
    p_.nTest = 80 - p_.nTrain;
    
  otherwise
    error(sprintf('unrecognized experiment name: %s', p_.experiment));
end

p_.downsample = 4; 

p_.gabor.M = p_.sz(1);        % see Gabor_construct.m
p_.gabor.b = p_.gabor.M / 8;  % see  " "
p_.gabor.sigma = p_.gabor.b;  % see  " "

p_.sift.subsamp = 1;          % see sift_macrofeatures.m
p_.sift.step = p_.downsample; % see  " "
p_.sift.macrosl = 1;          % see  " "
                              % Note: don't use macrofeatures
                              % since there is no sparse coding

p_.wavelet.J = 4;             % see wavelet_feature.m

p_.rootDir = fullfile('Outputs', [p_.experiment '_seed' num2str(p_.seed)]);

rng(p_.seed);
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
downsample_rows = @(I,p) I(1:p:end,:,:,:);
downsample_cols = @(I,p) I(:,1:p:end,:,:);
downsample = @(I,p) downsample_cols(downsample_rows(I,p),p);

% Note: making these intermediate lambda functions introduces some
% inefficiency.  However, for now we don't worry about this.
run_sift = @(I) sift_macrofeatures(single(I), ... 
                                   'subsamp', p_.sift.subsamp, ...
                                   'step', p_.downsample, ...
                                   'macrosl', p_.sift.macrosl);

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);
run_gabor = @(I) downsample(Gabor_transform(I, G), p_.downsample);

run_wavelet = @(I) wavelet_feature(I, p_.wavelet.J);



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
        loadedRawImagesYet=true;
        
        % REMOVE THIS AFTER DONE TESTING!!!
        if 0
            yAll = unique(data.y);
            idx = (data.y == yAll(1)) | (data.y == yAll(2));
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
        
        tic
        fprintf('[%s]: saving SIFT features...\n', mfilename);
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
        
        tic
        fprintf('[%s]: saving Gabor features...\n', mfilename);
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
       
        tic
        fprintf('[%s]: saving wavelet features...\n', mfilename);
        save(fnWavelet, 'feats', '-v7.3');
        clear feats;
        toc
    end
end



%% Pooling and classification

wi_sos_pool = @(X, k) spatial_pool(X, 'sos', k);
wi_mf_pool = @(X, k) spatial_pool(X, 'fun', k);

% TODO: re-enable wavelet features once they are the same
% dimensions as the other feature maps.
%featFiles = {siftFn, gaborFn, waveletFn};
featFiles = {siftFn, gaborFn};

for ii = 1:length(experimentDir), eDir = experimentDir{ii};
    fprintf('[%s]: starting train/test split %d (of %d)\n', ...
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
        % For now, we are only doing "whole image" pooling.  
        % Could use spatial_pool() to implement grid pooling...
        %
        poolfuncs = { @(X) spatial_pool(X, 'avg'), ...
                      @(X) spatial_pool(X, 'max'), ...
                      @(X) spatial_pool(X, 'pnorm', 2), ...
                      @(X) spatial_pool(X, 'fun', pMF), ...
                      @(X) spatial_pool(X, 'sos', pSOS)};

        Yhat = zeros(numel(feats.test.y), length(poolfuncs)+1);
        Yhat(:,1) = feats.test.y;
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

for ii = 1:length(experimentDir), eDir = experimentDir{ii};
    % TODO  aggregate across folds and report results
end


diary off;