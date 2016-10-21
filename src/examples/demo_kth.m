% This script assumes you are runnign from pwd.

rng(9999, 'twister');


%% load data
n_folds = 5;

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
fun_pooling = @(X) spatial_pool(X, 'fun', 12);  % TODO: hyperparameter selection

f_pool = {max_pooling, avg_pooling, avg_abs_pooling, ell2_pooling, fun_pooling};


%% simple analysis

Y_hat = [];
Y_true = [];

for ii = 1:n_folds
    test_id = ii;
    valid_id = ii+1;  if valid_set > n_folds, valid_set = 1; end
    train_id = setdiff(1:5, [test_set, valid_set]);
    
    is_train = ismember(data.fold, train_id);
    is_valid = ismember(data.fold, valid_id);
    is_test = ismember(data.fold, test_id);

    for ff = 1:length(f_feat)
        for pp = 1:length(f_pool)
            % recalculating features this way is computationally wasteful 
            % but uses minimal memory resources.
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
                Y_true = zeros(size(Y_hat));
            end
            Y_hat(:, ff, pp, ii) = y_hat(:);
            Y_true(:, ff, pp, ii) = y_test(:);
        end
    end
end

return % TEMP


X_vec = reshape(data.X, size(data.X,1)*size(data.X,2), size(data.X, 3));
x_avg = mean(X_vec, 1);
x_max = max(X_vec, [], 1);
x_ell2 = sum(X_vec.^2);

figure;
boxplot(x_avg, data.y);
title('KTH TIPS: average pooling (whole image)');

figure;
boxplot(x_max, data.y);
title('KTH TIPS: max pooling (whole image)');

figure;
boxplot(x_ell2, data.y);
title('KTH TIPS: ell^2 pooling (whole image)');

