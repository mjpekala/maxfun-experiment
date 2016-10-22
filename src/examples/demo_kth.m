% DEMO_KTH  Simple pooling experiments on a texture data set.
%
% This script assumes you are runnign from pwd.

% mjp, oct 2016

rng(9999, 'twister');


%% load data
n_folds = 3;

data = load_image_dataset('../datasets/KTH_TIPS', [200 200]);
data.X = single(data.X);

nth = @(x,n) x(n);  % due to matlab slicing limitation
get_class_name = @(fn) nth(strsplit(fn, '/'), 4);

y_all = sort(unique(data.y));
y_str = cellfun(get_class_name, data.files);

data.fold = assign_folds(data.y, n_folds);


%% set up feature extractors
p_.sift.size = 4;
p_.sift.geom = [4 4 8];      % [nX nY nAngles]

sift_feats = @(I) dsift2(I, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

p_.gabor.M = size(data.X,1);
p_.gabor.b = p_.gabor.M / 12; 
p_.gabor.sigma = p_.gabor.b; 

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);

% limit # of gabor features to be the same as SIFT
G = G(:,:,1:prod(p_.sift.geom));

gabor_feats = @(I) abs(Gabor_transform(I, G));

f_feat = {sift_feats, gabor_feats};


%% set up pooling functions

max_pooling = @(X) spatial_pool(X, 'max');
avg_pooling = @(X) spatial_pool(X, 'avg');
avg_abs_pooling = @(X) spatial_pool(abs(X), 'max');
ell2_pooling = @(X) spatial_pool(X, 'pnorm', 2);
fun_pooling = @(X) spatial_pool(X, 'fun', 12);  % *** TODO: hyperparameter selection

f_pool = {max_pooling, avg_pooling, avg_abs_pooling, ell2_pooling, fun_pooling};


%% do some very simple analysis

y_hat_all = {};
y_true_all = {};

%parfor fold_id = 1:n_folds
for fold_id = 1:n_folds
    
    % partition data into subsets.
    % note that the size of the test set may vary from fold to fold.
    test_id = fold_id;
    valid_id = fold_id+1;  if valid_id > n_folds, valid_id = 1; end
    train_id = setdiff(1:5, [test_id, valid_id]);
    
    is_train = ismember(data.fold, train_id);
    is_valid = ismember(data.fold, valid_id);
    is_test = ismember(data.fold, test_id);
   
    Y_hat = zeros(sum(is_test), length(f_feat), length(f_pool));

    for ff = 1:length(f_feat)
        for pp = 1:length(f_pool)
            % recalculating features each time is computationally wasteful but 
            % this avoids having to keep large datasets in memory.
            fprintf('[%s]: creating features (f=%d, p=%d)\n', mfilename, ff, pp);
            f = @(I) f_pool{pp}(f_feat{ff}(I));
 
            tic; 
            X_train = squeeze(map_image(data.X(:,:,is_train), f)); 
            y_train = data.y(is_train);
            toc

            tic; 
            X_test = squeeze(map_image(data.X(:,:,is_test), f)); 
            y_test = data.y(is_test);
            toc

            % the transpose below is for rows-as-objects
            [y_hat, metrics] = eval_svm(X_train', y_train, X_test', y_test);

            if isempty(Y_hat)
                Y_hat = zeros(numel(y_hat), length(f_feat), length(f_pool), n_folds);
                Y_true = zeros(numel(y_test), n_folds);
            end
            Y_hat(:, ff, pp) = y_hat(:);
        end
    end

    y_hat_all{fold_id} = Y_hat;
    y_true_all{fold_id} = y_test;

    save(sprintf('results_kth_fold%02d.mat', fold_id), 'Y_hat', 'y_test');

    recall_per_class(Y_hat, y_test);
    
    mcnemar_multiclass(Y_hat(:,1,4), Y_hat(:,2,5), y_test, 'SIFT+L2', 'Gabor_MF');

    break % for now, one fold only
end

return % TEMP
