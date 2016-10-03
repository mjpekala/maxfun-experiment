%  DEMO_MULTICLASS_WINDOWED  
%
%   A variant of demo_multiclass_whole_image that partitions the
%   images into multiple "windows" prior to pooling.
%


% mjp, Oct. 2016


%% Experiment Parameters 

p_.seed = 9999;
%p_.nSplits = 10;  
p_.nSplits = 5;    % TEMP

%p_.classesToUse = 1:101;
p_.classesToUse = [1 2 3 4 6 13 20 24 48 56 95];


% --= Dataset Parameters =--
% NOTE: some of the paths below assume you are running this script from pwd.
%       Make sure to change as appropriate.
p_.imageDir = '../datasets/101_ObjectCategories';
p_.sz = [200 200];    % Gabor feature code requires square images
p_.window_dim = 25;   % set to 0 for whole image pooling
%p_.window_dim = 0;   % set to 0 for whole image pooling
p_.downsample = 4;    % feature space downsampling; alleviates memory issues
p_.nTrain = 30;       
p_.nTest = 30;       

% --= Gabor parameters =--
% choose b s.t., given p_.sz, Gabor has ~128 feature dimensions
p_.gabor.M = p_.sz(1);
p_.gabor.b = p_.gabor.M / 10;
p_.gabor.sigma = p_.gabor.b; 

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);

% --= SIFT parameters =--
p_.sift.size = 4;             % see dl_vsift


% --= output path stuff =--
p_.experiment = sprintf('caltech-%d-w%d', numel(p_.classesToUse), p_.window_dim);
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
feature_type_names = {'SIFT', 'Gabor'};

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
experimentDir = {};

for splitId = 1:p_.nSplits
    experimentDir{splitId} = fullfile(p_.rootDir, sprintf('split_%0.2d', splitId));
    make_dir(experimentDir{splitId});
    
    if exist(fullfile(experimentDir{splitId}, p_.fn.results))
        fprintf('[%s]: skipping split %d (already calculated)\n', ...
                mfilename, splitId);
        continue;  % already completed this train/test split
    end

    %% determine the train/test split
    % Note this will be the same split for all feature types.
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
    n_pool_strat = 5;
    Yhat = zeros(numel(test.y), n_pool_strat, numel(feature_algos));  

    for algoId = 1:numel(feature_algos)
        %------------------------------
        % generate features
        %------------------------------
        if p_.window_dim > 0
            % compose the feature generation with image partitioning
            f_algo = @(I) pooling_regions(feature_algos{algoId}(I), p_.window_dim, true);
        else
            % whole image pooling
            f_algo = feature_algos{algoId};
        end
    
        fprintf('\n\n');   
        fprintf('[%s]: generating training data for %s\n', mfilename, feature_type_names{algoId});
        feats.train.X = map_image(train.I, f_algo);
        feats.train.y = train.y;
        
        fprintf('[%s]: generating test data for %s\n', mfilename, feature_type_names{algoId});
        feats.test.X = map_image(test.I, f_algo);
        feats.test.y = test.y;


        fprintf('[%s]: train size: %s\n', mfilename, num2str(size(feats.train.X)));
        fprintf('[%s]: test size:  %s\n', mfilename, num2str(size(feats.test.X)));
 
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
        % (optional) study maxpool properties for these features
        %------------------------------
        if splitId == 1 && 1
            %study_maxpool_support(feats.train.X(:,:,[1 10 100],:));
            study_maxpool_support(feats.train.X);
            drawnow;
            fn = fullfile(experimentDir{splitId}, ...
                          sprintf('mp_study_%d.fig', algoId));
            title(sprintf('maxfun analysis for feature type %d', algoId));
            saveas(gca, fn)
        end
       
        %------------------------------
        % evaluate each candidate pooling function
        %------------------------------
        poolfuncs = {max_pooling, avg_pooling, avg_abs_pooling, ell2_pooling, maxfun_pooling};
        for ii = 1:length(poolfuncs)
            fprintf('[%s]: evaluating strategy (%d,%d)  (of %d, %d)\n', ...
                    mfilename, algoId, ii, numel(feature_algos), length(poolfuncs));
            
            pStartTime = tic;
            Xtrain = poolfuncs{ii}(feats.train.X);
            Xtest = poolfuncs{ii}(feats.test.X);
            poolTime = toc(pStartTime);
            
            % transpose because the SVM codes want objects-as-rows.
            [yHat, metrics] = eval_svm(Xtrain', feats.train.y, Xtest', feats.test.y);
            svmTime = toc(pStartTime) - poolTime;
            Yhat(:, ii, algoId) = yHat;
            fprintf('    mean acc: %0.2f\n', mean(metrics.acc));
            fprintf('    runtime (sec); pool = %0.2f, SVM = %0.2f\n', ...
                    poolTime, svmTime);

            clear Xtrain Xtest;
        end
    end

    % save data to file
    fn = fullfile(experimentDir{splitId}, p_.fn.results);
    save(fn, 'train', 'test', 'p_', 'Yhat', '-v7.3');

    clear feats;
 
    fprintf('[%s]: finished split %d; net time %0.2f (min)\n', ...
            mfilename, splitId, toc(overallTimer)/60.);
    
    for ii = 1:size(Yhat,3)
        is_correct = bsxfun(@eq, Yhat(:,:,ii),  test.y(:));
        acc = sum(is_correct,1) / size(is_correct,1);
        fprintf('   acc. for algo. %d: %s\n', ii, num2str(100*acc));
    end
    
end


%% Post-processing / analysis
%
% This is simply aggregating results across the various train/test splits.

for kk = 1:length(experimentDir)
    % load results from this split
    eDir = experimentDir{kk};
    fn = fullfile(experimentDir{kk}, p_.fn.results);
    load(fn);
  
    if kk == 1
        % Construct matrices to hold aggregate results
        yAll = sort(unique(test.y));
        n_classes = length(unique(yAll));
        n_pool_strat = size(Yhat,2);
        n_feat_types = size(Yhat,3);
        Acc = zeros(n_classes, n_pool_strat, p_.nSplits, n_feat_types);
    end

    % add results for this particular split
    for jj = 1:n_pool_strat
        for ll = 1:n_feat_types
            for ii = 1:length(yAll)
                idx = (test.y == yAll(ii));
                is_correct = Yhat(idx, jj, ll) == test.y(idx);
                Acc(ii,jj,kk,ll) = sum(is_correct) / length(is_correct);
            end
        end
    end
end


% display summary statistics
for ll = 1:size(Acc,4)
    Acc_feat = Acc(:,:,:,ll);
    Acc_mean = mean(Acc_feat,3);
    Acc_std = std(Acc_feat,0,3);
 
    header = '       | ';
    for jj = 1:size(Acc_mean,2)
        header = [header, sprintf(' pool strat %02d  | ', jj)];
    end
   
    fprintf('%s\n', header);
    fprintf('-------+------------------------------------------------------------------------------------------\n');
    for ii = 1:size(Acc_mean,1)
        fprintf('y=%3d  |', yAll(ii));
        for jj = 1:size(Acc_mean,2)
            fprintf(' %05.2f +/- %05.2f |', 100*Acc_mean(ii,jj), 100*Acc_std(ii,jj));
        end
        fprintf('\n');
    end
    
    % also show a marginal
    fprintf('-------+------------------------------------------------------------------------------------------\n');
    fprintf(' avg   |');
    for jj = 1:size(Acc_mean,2)
        fprintf(' %05.2f +/- %05.2f |', ...
                100*mean(Acc_mean(:,jj)), 100*mean(Acc_std(:,jj)));
    end
    fprintf('\n\n\n');
end



diary off;
