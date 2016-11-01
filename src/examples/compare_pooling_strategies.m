% Compares various pooling strategies.
%
%  Note: be sure to run setup.m before calling this script.
%  

% mjp, oct 2016


%% Experiment Parameters

% Dataset parameters
if 0
    p_.data_dir = '../datasets/101_ObjectCategories';
    p_.classes_to_use = [];       % empty := use whole data set
    p_.classes_to_use = [10 11];  % a quick test case
else
    p_.data_dir = '../datasets/curetgrey';
    p_.classes_to_use = [4 8 10 24 35];  
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
sift_xform = @(X) dsift2(X, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

gabor_xform = @(X) abs(Gabor_transform_ns(X, p_.gabor.A, p_.gabor.B));

f_feat = {sift_xform, gabor_xform};

f_pool = { @(X) spatial_pool(X, 'max'), 
           @(X) spatial_pool(X, 'avg'),
           @(X) spatial_pool(X, 'pnorm', 2), 
           @(X) spatial_pool(X, 'maxfun', [3:8]),
           @(X) spatial_pool(X, 'maxfun', [15:20]),
           @(X) spatial_pool(X, 'maxfun', [70:75]) };



%% Generate features

% Note: this only needs to be done once, assuming the parameters
%       have not changed. If you do change parameters, delete
%       this file and re-run.

feat_file = 'pooled_features.mat';

if exist(feat_file)
    fprintf('[%s]: feature file %s already exists; skipping to analysis\n', ...
            mfilename, feat_file);
else
    % load data and crop (if needed) to ensure dimension is even.
    % "cropping" consists of dropping at most one row and column.
    data = load_image_dataset(p_.data_dir);
    for ii = 1:length(data.y)
        if mod(size(data.X{ii},1), 2) == 1
            data.X{ii} = data.X{ii}(1:end-1,:);
        end
        if mod(size(data.X{ii},2), 2) == 1
            data.X{ii} = data.X{ii}(:,1:end-1);
        end
    end

    % reduce data set (optional)
    if ~isempty(p_.classes_to_use)
        slice = ismember(data.y, p_.classes_to_use);
        data.y = data.y(slice);
        data.files = data.files(slice);
        data.X = data.X(slice);
        fprintf('[%s]: reduced dataset has %d examples\n', mfilename, numel(data.y));
    end

    % make sure features are all of same dimension
    feat_dim = prod(p_.sift.geom);
    assert(feat_dim == p_.gabor.A * p_.gabor.B);

    % preallocate space for the processed data
    data.Xf = zeros(feat_dim, length(data.y), length(f_feat), length(f_pool));
    maxfun_sz = zeros(feat_dim, length(data.y), length(f_feat));
   
    % Process images one at a time (to conserve memory).
    timer = tic;   last_notify = 0;
    
    for ii = 1:length(data.y)
        for jj = 1:length(f_feat)
            % feature extraction just 1x for each object
            Xi = f_feat{jj}(data.X{ii});
            
            % do pooling
            for kk = 1:length(f_pool)-1
                data.Xf(:,ii,jj,kk) = f_pool{kk}(Xi);
            end
            
            % XXX - we handle maxfun separately for now (so can grab stats)
            % If you don't care about the statistics, can fold this back into
            % the above for loop.
            [data.Xf(:,ii,jj,end), nfo] = f_pool{end}(Xi);
            maxfun_sz(:,ii,jj) = nfo.w;
        end

        % provide status updates 
        elapsed = toc(timer) / 60;
        if elapsed > (2 + last_notify)
            fprintf('[%s]: processed %d examples (of %d) in %.2f min\n', mfilename, ii, length(data.y), elapsed);
            last_notify = elapsed;
        end
    end

    save(feat_file, 'data', 'p_', 'maxfun_sz');

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
n_train = 30;              % # of objects to use for training
n_test = 30;               %   "   " testing
n_splits = 10;

rng(9999, 'twister');

% load previously computed features
load(feat_file);


for split_id = 1:n_splits
    fprintf('[%s]: Starting classification experiment %d (of %d)\n', ...
            mfilename, split_id, n_splits);
    
    % note: we will use the exact same train/test items for each
    %       feature/pooling pair.
    [is_train, is_test] = select_n(data.y, n_train, n_test);
    
    y_train = data.y(is_train);
    y_test = data.y(is_test);
   
    % allocate memory
    if split_id == 1
        y_hat = zeros(sum(is_test), size(data.Xf,3), size(data.Xf,4), n_splits);
        y_true = zeros(sum(is_test), n_splits);
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

est_file = 'estimates.mat';
save(est_file, 'data', 'p_', 'y_hat', 'y_true');

% show recall rates
fprintf('\n            SIFT    \n---------+----------------------------------------\n')
[recall_sift, y_id_sift] = recall_per_class(squeeze(y_hat(:, 1, :, :)), y_true);
fprintf('\n            Gabor   \n---------+----------------------------------------\n')
[recall_gabor, y_id_gabor] = recall_per_class(squeeze(y_hat(:, 2, :, :)), y_true);

