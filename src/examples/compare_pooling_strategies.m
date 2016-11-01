% Compares various pooling strategies.
%
%  Note: be sure to run setup.m before calling this script.
%  

% mjp, oct 2016

%% Experiment Parameters

p_.data_dir = '../datasets/101_ObjectCategories';

% Parameters for SIFT
p_.sift.size = 4;
p_.sift.geom = [4 4 8];      % [nX nY nAngles]

sift_xform = @(X) dsift2(X, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

% Parameters for Gabor 
p_.gabor.A = 16;
p_.gabor.B = 8;

gabor_xform = @(X) Gabor_transform_ns(X, p_.gabor.A, p_.gabor.B);


% pooling strategies to compare
f_pool = { @(X) spatial_pool(X, 'max'), 
           @(X) spatial_pool(X, 'avg'),
           @(X) spatial_pool(X, 'pnorm', 2), 
           @(X) spatial_pool(X, 'maxfun', 5:5:30) };

% feature strategies to compare
f_feat = {sift_xform, gabor_xform};


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
    data = load_image_dataset(p_.data_dir);
    for ii = 1:length(data.y)
        if mod(size(data.X{ii},1), 2) == 1
            data.X{ii} = data.X{ii}(1:end-1,:);
        end
        if mod(size(data.X{ii},2), 2) == 1
            data.X{ii} = data.X{ii}(:,1:end-1);
        end
    end
   
    
    % TEMP TEMP TEMP - just for a small debugging experiment!!
    f_feat = f_feat(1);
    slice = 1:300;
    data.y = data.y(slice);
    data.files = data.files(slice);
    data.X = data.X(slice);
    % END TEMP TEMP TEMP
    
    
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
            % TEMP - we handle maxfun separately for now (so can grab stats)
            [data.Xf(:,ii,jj,kk), nfo] = f_pool{end}(Xi);
            maxfun_sz(:,ii,jj) = nfo.w;
        end

        % provide status updates 
        elapsed = toc(timer) / 60;
        if elapsed > (2 + last_notify)
            fprintf('[%s]: processed %d examples in %.2f min\n', mfilename, ii, elapsed);
            last_notify = elapsed;
        end
    end

    save(feat_file, 'data', 'p_');

    figure; hist(maxfun_sz(:));
    title('distribution of maxfun pooling sizes');
end




%% Analysis

%load(feat_file);
