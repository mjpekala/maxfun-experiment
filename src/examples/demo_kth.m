% DEMO_KTH  Simple pooling experiments on a texture data set.
%
% This script assumes you are runnign from pwd.

% mjp, oct 2016

rng(9999, 'twister');

% XXX: note to self - we probably want "mixed" images with both
% texture and edges.  The gabor will hit the textures and hopefully
% maxfun can help localize.  Hence, in this particular experiment we
% probably don't expect a bit difference between maxfun and avg.
%
% Given a large uniform texture there is no real reason to believe
% that honing in on a subset of it will be particularly advantageous.


%% load data
n_trials = 10;
p_.sz = 50;

data = load_image_dataset('../datasets/KTH_TIPS', [p_.sz p_.sz]); 
data.X = single(data.X);



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
avg_abs_pooling = @(X) spatial_pool(abs(X), 'avg');
ell2_pooling = @(X) spatial_pool(X, 'pnorm', 2);
fun_pooling = @(X) spatial_pool(X, 'fun', floor(p_.sz/4));  % *** TODO: hyperparameter selection

f_pool = {max_pooling, avg_pooling, avg_abs_pooling, ell2_pooling, fun_pooling};


%% do some very simple analysis

desc = sprintf('demo_kth_d=%d', p_.sz);
fprintf('[%s]: starting experiment "%s"\n', mfilename, desc);

diary(sprintf('log_%s_%s.txt', desc, datestr(now)));
main_timer = tic;

acc_all = zeros(length(unique(data.y)), length(f_feat), length(f_pool), n_trials);

for trial_id = 1:n_trials
    fprintf('-------------------------------------------------------\n');
    fprintf('[%s]: starting trial %d (of %d)\n', mfilename, trial_id, n_trials);
    fprintf('-------------------------------------------------------\n');

    % choose a random train/test split
    fold_id = assign_folds(data.y, 2);
    is_train = (fold_id == 1);
    is_test = (fold_id == 2);
    
    Y_hat = zeros(sum(is_test), length(f_feat), length(f_pool));

    for ff = 1:length(f_feat)
        for pp = 1:length(f_pool)
            % recalculating features each time is computationally wasteful but 
            % this avoids having to keep large datasets in memory.
            fprintf('[%s]: creating features (f=%d, p=%d)\n', mfilename, ff, pp);
            f = @(I) f_pool{pp}(f_feat{ff}(I));
 
            X_train = squeeze(map_image(data.X(:,:,is_train), f)); 
            y_train = data.y(is_train);

            X_test = squeeze(map_image(data.X(:,:,is_test), f)); 
            y_test = data.y(is_test);

            % train and test classifier.
            % the transpose below is for rows-as-objects
            [y_hat, metrics] = eval_svm(X_train', y_train, X_test', y_test);
            Y_hat(:, ff, pp) = y_hat(:);
            acc_all(:,ff,pp,trial_id) = metrics.acc;
        end
    end

    % store results from this trial
    save(sprintf('results_%s_trial=%02d.mat', desc, trial_id), 'Y_hat', 'y_test', 'p_');
   
    y_hat_all{trial_id} = Y_hat;
    y_true_all{trial_id} = y_test;

    % report results for this trial
    for ff = 1:length(f_feat)
        fprintf('[%s]: classification performance for feature type %d\n',  mfilename, ff);
        recall_per_class(Y_hat(:,ff,:,:), y_test, data.class_names);
    end
    mcnemar_multiclass(Y_hat(:,1,4), Y_hat(:,2,5), y_test, 'SIFT+L2', 'Gabor_MF');
    
    fprintf('[%s]: net time elapsed: %0.2f (min)\n', mfilename, toc(main_timer)/60);
end


% Analysis of aggregate performance.
for ff = 1:length(f_feat)
    fprintf('Feature type %d:\n', ff);
    
    for yi = 1:size(acc_all,1)
        fprintf('%14s |', data.class_names{yi});
        for pp = 1:length(f_pool)
            mu = 100*mean(acc_all(yi, ff, pp, :));
            sigma = 100*std(acc_all(yi, ff, pp, :));
            fprintf(' %6.2f (%5.2f) |', mu, sigma);
        end
        fprintf('\n');
    end
    fprintf('\n');
end


diary off;
