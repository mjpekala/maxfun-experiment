% Compares various "whole image" pooling strategies.
%
%  In this script we combine feature generation and pooling into
%  a single operation that we apply to all examples in a data set.
%  This approach has pros and cons:
%   + there are no issues with memory since we pool the (relatively
%     large) feature tensors immediately.  This also means we can
%     avoid downsampling.
%   - this makes it very difficult to do any kind of hyperparameter
%     optimization for pooling functions (e.g. choosing p for Lp
%     norms or minimum support sizes for maxfun).
% 
%  The assumption here is that the dataset of interest consists of
%  images stored as separate files in subdirectories and that the
%  subdirectory name represents the class label.  For example, the
%  Caltech-101 dataset follows this convention.  See
%  load_image_dataset() for more details.

% mjp, oct 2016


%% Experiment Parameters

% Dataset parameters
if 1
    p_.desc = 'caltech101_small';
    p_.data_dir = '../datasets/101_ObjectCategories';
    p_.classes_to_use = [];    
else
    p_.desc = 'CURET';
    p_.data_dir = '../datasets/curetgrey';
    %p_.classes_to_use = [4 8 10 24 35];  
    p_.classes_to_use = [4 8];
end


% Parameters for SIFT
p_.sift.size = 4;
p_.sift.geom = [4 4 8];       % [nX nY nAngles]

% Parameters for Gabor 
p_.gabor.A = 16;
p_.gabor.B = 8;

% Parameters for maxfun
% (hardcoded - see f_pool below)


%% Helper functions

% This is dense sift with no downsampling (step = 1)
sift_xform = @(X) dsift2(X, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

gabor_xform = @(X) abs(Gabor_transform_ns(X, p_.gabor.A, p_.gabor.B));

% the feature transforms we consider are encoded as function
% handles in a cell array.
f_feat = {sift_xform, gabor_xform};

% pooling functions are similarly stored in a cell array
f_pool = { @(X) spatial_pool(X, 'max'), 
           @(X) spatial_pool(X, 'avg'),
           @(X) spatial_pool(X, 'pnorm', 2), 
           @(X) spatial_pool(X, 'maxfun', [3:6]),
           @(X) spatial_pool(X, 'maxfun', [15:18]),
           @(X) spatial_pool(X, 'maxfun', [70:73]) };



%% Generate features

% Note: this only needs to be done once, assuming the parameters
%       have not changed. If you do change parameters, delete
%       the preprocessed data file and re-run the script.

feat_file = [p_.desc  '_features.mat'];

if exist(feat_file)
    fprintf('[%s]: feature file %s already exists; skipping to analysis\n', ...
            mfilename, feat_file);
else
    p_   % show parameters to user
    
    % First, make sure Gabor and SIFT will have the same # of
    % features (to facilitate fair comparison)
    n_feats = prod(p_.sift.geom);
    assert(n_feats == p_.gabor.A * p_.gabor.B);
    
    % load data and crop (if needed) to ensure dimension is even.
    % "cropping" consists of dropping at most one row and column.
    % (at one point this was a requirement of the Gabor codes)
    data = load_image_dataset(p_.data_dir);
    for ii = 1:length(data.y)
        if mod(size(data.X{ii},1), 2) == 1
            data.X{ii} = data.X{ii}(1:end-1,:);
        end
        if mod(size(data.X{ii},2), 2) == 1
            data.X{ii} = data.X{ii}(:,1:end-1);
        end
    end

    % distill down to the set of classes of interest (optional)
    if ~isempty(p_.classes_to_use)
        slice = ismember(data.y, p_.classes_to_use);
        data.y = data.y(slice);
        data.files = data.files(slice);
        data.X = data.X(slice);
    end
    

    % preallocate space for the processed data
    data.Xf = zeros(n_feats, length(data.y), length(f_feat), length(f_pool));  
    maxfun_sz = -1*ones(n_feats, length(data.y), length(f_feat));  % support size used by maxfun
   
    fprintf('[%s]: After preprocesing data set has %d examples and %d classes\n', mfilename, length(data.y), length(unique(data.y)));
    fprintf('[%s]: Feature dimension will be: %d\n', mfilename, n_feats);
    
    
    % Process images one at a time (to conserve memory).
    timer = tic;   last_notify = -Inf;
    
    for ii = 1:length(data.y)
        for jj = 1:length(f_feat)
            % feature extraction 
            Xi = f_feat{jj}(data.X{ii});
            
            % pooling
            for kk = 1:length(f_pool)
                data.Xf(:,ii,jj,kk) = f_pool{kk}(Xi);
            end
           
            % Special case processing for maxfun; if you don't care about this
            % analysis you can turn this code off (does not impact classification 
            % results in any way)
            %[~, nfo] = spatial_pool(Xi, 'maxfun', [3:8]);
            %maxfun_sz(:,ii,jj) = nfo.w;
        end

        % provide status updates 
        elapsed = toc(timer) / 60;
        if elapsed > (2 + last_notify)
            fprintf('[%s]: processed %d examples (of %d) in %.2f min\n', mfilename, ii, length(data.y), elapsed);
            last_notify = elapsed;
        end
    end

    runtime = toc(timer);
    save(feat_file, 'data', 'p_', 'maxfun_sz', 'runtime', '-v7.3');
    fprintf('[%s]: done.  Took %d minutes total\n', mfilename, runtime);

    % support size analysis
    %for ii = 1:size(maxfun_sz,3)
    %    fprintf('[%s]: maxfun support frequencies for feature type %d:\n', mfilename, ii);
    %    mf_stats = zeros(numel(p_.mf_supp),1);
    %    tmp = maxfun_sz(:,:,ii);
    %    for ii = 1:numel(mf_stats)
    %        mf_stats(ii) = sum(tmp(:) == p_.mf_supp(ii));
    %        fprintf('    %d : %d\n', p_.mf_supp(ii), mf_stats(ii));
    %    end
    %end
end




%% Classification

% Classification parameters
n_train = 30;              % # of objects to use for training from each class
n_test = 30;               %   "   "  testing  "  "
n_splits = 10;             % # of train/test splits to perform

rng(9999, 'twister');

% load previously computed features
load(feat_file);


for split_id = 1:n_splits
    fprintf('[%s]: Starting classification experiment %d (of %d)\n', ...
            mfilename, split_id, n_splits);
  
    % generate a train/test split 
    % (will be used for all feature/pooling pairs)
    [is_train, is_test] = select_n(data.y, n_train, n_test);
    assert(sum(is_train & is_test) == 0);
    
    y_train = data.y(is_train);
    y_test = data.y(is_test);
   
    % allocate memory (now that test set sizes are known)
    if split_id == 1
        y_hat = zeros(sum(is_test), size(data.Xf,3), size(data.Xf,4), n_splits);
        y_true = zeros(sum(is_test), n_splits);
        
        % some diagnostics (related to data set sizes)
        y_all = unique(data.y)
        fprintf('  y    |  #train  |  #test  |  # in dataset \n-------+----------+---------+-----------------\n')
        for ii = 1:length(y_all)
            fprintf('  %3d  |   %4d   |   %4d  |  %4d (%4.2f%%)\n', y_all(ii), sum(data.y(is_train) == y_all(ii)), sum(data.y(is_test) == y_all(ii)), sum(data.y == y_all(ii)), 100 * sum(data.y==y_all(ii)) / numel(data.y));
        end
    end
    
    y_true(:,split_id) = y_test;
    
    for ii = 1:size(data.Xf,3)
        for jj = 1:size(data.Xf,4)
            X_train = data.Xf(:,is_train,ii,jj);
            X_test = data.Xf(:,is_test,ii,jj);
            [y_hat_ij, metrics] = eval_svm(X_train', y_train, X_test', y_test);
            y_hat(:,ii,jj,split_id) = y_hat_ij;
        end
    end
end

est_file = [p_.desc '_estimates.mat'];
save(est_file, 'data', 'p_', 'y_hat', 'y_true', '-v7.3');

% show recall rates
for ii = 1:size(y_hat, 2)
    fprintf('\n            Feature Type %d    \n---------+---------------------------------------------------------------\n', ii)
    [recall, y_id] = recall_per_class(squeeze(y_hat(:, ii, :, :)), y_true);
end

