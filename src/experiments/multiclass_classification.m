%  MULTICLASS_CLASSIFICATION  
%
%   Compare the impact of different features and pooling strategies on
%   one-vs-all image classification problems.  Originally this code
%   started as a way to repeat/extend the experiments of [1].
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
p_.nSplits = 2;  % FWIW, in section 2.3 in [1], the authors use 10 splits.
p_.nAtoms = 128;  % set to a non-positive value if you don't want
                  % sparse coded features.

switch lower(p_.experiment)
  case {'caltech-101', 'caltech101'}
    p_.imageDir = '/Users/pekalmj1/Data/caltech_101/101_ObjectCategories';
    p_.sz = [200 300];    % assume fixed-size inputs in this script
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

p_.rootDir = fullfile('Outputs', [p_.experiment '_' num2str(p_.seed)]);

rng(p_.seed);
timestamp = datestr(now);
overallTimer = tic;

p_  % show parameters to user


%% some helper functions

shuffle = @(x) x(randperm(numel(x)));
make_dir = @(dirName) ~exist(dirName) && mkdir(dirName);

run_sift = @(I) sift_macrofeatures(I, 'subsamp', 1, 'step', 4, 'macrosl', 2);
featurize = @(fn) run_sift(read_images(fn, sz));

% take a tensor with dimensions:   (rows, cols, #features, #examples)
% and reshapes into a matrix:      (#features, #objects)
tensor_to_matrix = @(T) reshape(permute(T, [3 1 2 4]), size(T,3), size(T,1)*size(T,2)*size(T,4));

% un-does the action of tensor_to_matrix
matrix_to_tensor = @(M, sz) permute(reshape(M, size(M,1), sz(1), sz(2), sz(4)), [2 3 1 4]);



%% Pre-genenerate train/test splits for each experiment
%
% We do it this way in order to conserve memory.  Some datasets are
% sufficiently large that we can't featurize the entire data set in
% memory at once.

make_dir(p_.rootDir);
diary(fullfile(p_.rootDir, ['log_one_vs_all_' timestamp '.txt']));

data = load_image_dataset(p_.imageDir, p_.sz);
for splitId = 1:p_.nSplits
    % create output file (if it does not already exist)
    experimentDir{splitId} = fullfile(p_.rootDir, sprintf('split_%0.2d', splitId));
    make_dir(experimentDir{splitId});
    fn = fullfile(experimentDir{splitId}, 'data.mat');
    if exist(fn, 'file')
        fprintf('[%s]: data file already exists for train/test split %d; re-using...\n', mfilename, splitId);
        continue;
    end
    
    fprintf('[%s]: Creating train/test split %d (of %d)\n', mfilename, splitId, p_.nSplits);
    [isTrain, isTest] = select_n(data.y, p_.nTrain, p_.nTest);

    train.X = data.X(:,:,isTrain);
    train.y = data.y(isTrain);
    train.files = data.files(isTrain);
    train.idx = find(isTrain);
 
    test.X = data.X(:,:,isTest);
    test.y = data.y(isTest);
    test.files = data.files(isTest);
    test.idx = find(isTest);

    % save data to file
    save(fn, 'train', 'test', 'p_', '-v7.3');
    clear train test isTrain isTest;
end

clear data;


%% Run Experiments

for ii = 1:length(experimentDir)
    fprintf('[%s]: Processing train/test split %d (of %d)\n', mfilename, ii, p_.nSplits);
    eDir = experimentDir{ii};
    
    load(fullfile(eDir, 'data.mat'), 'train', 'test');
    return % TEMP

    %% generate low-level features for training data
    Xraw = cellfun_status(featurize, imageFiles(isTrain), 'SIFT train');
    Xraw = cat(4, Xraw{:});
    Xraw = double(Xraw);
    train.y = y(isTrain);

 
    %% learn dictionary
    lambdaAll = [.0001 .001 .01];
    lambdaAll = [.0001];   % XXX: hardcoded choice
    dlTimer = tic;
    [D, param] = learn_dictionary(tensor_to_matrix(Xraw), y, ...
                                  'k', 2, 'lambdas', lambdaAll, 'nAtoms', nAtoms);
    fprintf('[%s]: dictionary learning took %0.2f (min)\n', mfilename, toc(dlTimer)/60);
 
    
    %% Sparse coding (encode train and test data)
    fprintf('[%s]: encoding data using dictionary\n', mfilename);

    % Note: the following encodes each example independently. 
    %
    scTimer = tic;
    train.X = zeros(size(Xraw,1), size(Xraw,2), size(D,2), size(Xraw,4));
    for ii = 1:size(Xraw, 4)
        Xi = Xraw(:,:,:,ii);
        [height, width, chan] = size(Xi);
        Xi = reshape(Xi, height*width, chan);
        Ai = full(mexLasso(Xi', D, param));
        Ai = reshape(Ai', height, width, size(D,2));
        train.X(:,:,:,ii) = Ai;
    end
    clear Xi Ai Xraw;
    fprintf('[%s]: sparse coding (train data) took %0.2f (sec)\n', mfilename, toc(scTimer));
   
    % now the test data
    %
    Xraw = cellfun_status(featurize, imageFiles(isTest), 'SIFT test');
    Xraw = cat(4, Xraw{:});
    Xraw = double(Xraw);
    test.y = y(isTest);
    
    scTimer = tic;
    test.X = zeros(size(Xraw,1), size(Xraw,2), size(D,2), size(Xraw,4));
    for ii = 1:size(Xraw, 4)
        Xi = Xraw(:,:,:,ii);
        [height, width, chan] = size(Xi);
        Xi = reshape(Xi, height*width, chan);
        Ai = full(mexLasso(Xi', D, param));
        Ai = reshape(Ai', height, width, size(D,2));
        test.X(:,:,:,ii) = Ai;
    end
    clear Xi Ai Xraw;
    fprintf('[%s]: sparse coding (test data) took %0.2f (sec)\n', mfilename, toc(scTimer));


    save(fullfile(experimentDir, 'data.mat'), 'train', 'test', 'D', 'param', 'isTrain', 'isTest', '-v7.3');
    fprintf('[%s]: saved features in "%s"\n', mfilename, experimentDir);

 
    %% Pooling
    
    % Step 1: pooling parameter selection (for those operations that need it)
    %
    wi_sos_pool = @(X, k) spatial_pool(X, 'sos', k);
    wi_mf_pool = @(X, k) spatial_pool(X, 'fun', k);
       
    % assign folds so that all pooling functions see the same data splits. 
    pFoldId = assign_folds(train.y, 5);
        
    pcvTimer = tic;
    pMF = select_scalar_pooling_param(train.X, train.y, wi_mf_pool, 4:2:20, pFoldId); 
        
    % XXX: whether we should select pSOS separately or base it off the
    % MAXFUN value depends upon whether we are evaluating this as
    % a separate methodology or merely viewing it as an upper
    % bound on MAXFUN.
    pSOS = select_scalar_pooling_param(train.X, train.y, wi_sos_pool, 6:2:24, pFoldId); 
    %pSOS = pMF;

    fprintf('[%s]: runtime for pooling parameter selection: %0.2f (min)\n', ...
            mfilename, toc(pcvTimer)/60.);

    % Step 2: pool the features
    %
    % This is for "whole image" pooling.  The features from all spatial
    % regions are combined to form a single feature vector for each
    % object.
    train.avgpool = spatial_pool(train.X, 'avg');
    train.funpool  = spatial_pool(train.X, 'fun', pMF);
    train.maxpool = spatial_pool(train.X, 'max');
    train.sospool = spatial_pool(train.X, 'sos', pSOS);  
    train.sqrtpool = spatial_pool(train.X, 'pnorm', 2);

    test.avgpool = spatial_pool(test.X, 'avg');
    test.funpool  = spatial_pool(test.X, 'fun', pMF);
    test.maxpool = spatial_pool(test.X, 'max');
    test.sospool = spatial_pool(test.X, 'sos', pSOS);  
    test.sqrtpool = spatial_pool(test.X, 'pnorm', 2);

    
    %% Classification

    %
    % Since the sample sizes are so small, we'll store the raw
    % results (rather than accuracy values).  Otherwise, aggregate
    % statistics may be a bit misleading.
    %

    % the transposes below are because svmtrain wants rows-as-examples
    [yHat, avgp.metrics] = eval_svm(train.avgpool', train.y, test.avgpool', test.y);
    [yHat, fun.metrics] = eval_svm(train.funpool', train.y, test.funpool', test.y);
    [yHat, sos.metrics] = eval_svm(train.sospool', train.y, test.sospool', test.y);
    [yHat, maxp.metrics] = eval_svm(train.maxpool', train.y, test.maxpool', test.y);
    [yHat, sqrt.metrics] = eval_svm(train.sqrtpool', train.y, test.sqrtpool', test.y);
    
    % some aggregate statistics
    S = 100*[avgp.metrics.acc fun.metrics.acc sos.metrics.acc maxp.metrics.acc sqrt.metrics.acc];
    mr = mean_rank(S);

    %% Report results for this split
    fprintf('[%s]: avgpool acc / mean rank:   %0.2f%% / %0.2f\n', ...
            mfilename, 100*avgp.metrics.nCorrect/numel(test.y), mr(1));
    fprintf('[%s]: funpool acc / mean rank:   %0.2f%% / %0.2f\n', ...
            mfilename, 100*fun.metrics.nCorrect/numel(test.y), mr(2));
    fprintf('[%s]: sospool acc / mean rank:   %0.2f%% / %0.2f\n', ...
            mfilename, 100*sos.metrics.nCorrect/numel(test.y), mr(3));
    fprintf('[%s]: maxpool acc / mean rank:   %0.2f%% / %0.2f\n', ...
            mfilename, 100*maxp.metrics.nCorrect/numel(test.y), mr(4));
    fprintf('[%s]: sqrtpool acc / mean rank:  %0.2f%% / %0.2f\n', ...
            mfilename, 100*sqrt.metrics.nCorrect/numel(test.y), mr(5));
    
    fprintf('%19s (cnt) : %8s  %8s  %8s  %8s  %8s : funrank\n', ...
            'class', 'avg', 'fun', 'sos', 'max', 'sqrt');
    for ii = 1:size(S,1)
        tr = tiedrank(100 - S(ii,:));
        fprintf('%20s (%2d) : %8.2f  %8.2f  %8.2f  %8.2f  %8.2f : %0.1f\n', ...
                classNames{ii}, sum(test.y==ii), S(ii,1), S(ii,2), S(ii,3), S(ii,4), S(ii,5), tr(2));
    end
   
    Sall = [Sall S];
    
    save(fullfile(experimentDir, 'perf.mat'), 'S', 'mr', 'avgp', 'fun', 'sos', 'maxp', 'sqrt', '-v7.3');
end


fprintf('[%s]: total runtime %0.2f (min)\n', mfilename, toc(overallTimer)/60.);
save(fullfile(experimentDir, 'Sall.mat'), 'Sall', '-v7.3');

diary off;