% CLASSIFY_IMAGES
%
%  --= For CIFAR-10 =--
%
%     Note that some of the weakest deep learning classifiers still get
%     ~75%  accuracy on CIFAR-10, e.g. see:
%         http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
%
%     I've seen some anectodal references to ~50% accuracy with a SVM; e.g. 
%        Martin & Vatsavai "Exploring Improvements for Simple Image Classification" 2013.
%        Le & Gopee "Classifying CIFAR-10 Images Using Unsupervised Feature & Ensemble Learning"
%
%     As of this writing, our "one layer" accuracy is relatively poor
%     (~40%); however, we have not exhausted all possible
%     pre-processing options.
%
%
%   --= For Caltech-101 Lean =--
%
%     This is a custom experiment, so there are not likely to be
%     existing baselines.  We will have to use our judgement in terms
%     of what is reasonable here (again, in a "one layer" context).
%

% mjp, oct 2017

rng(1066);

%% Experiment parameters
pc.feature_type = 'dyadic-edge';
pc.alpha_all = linspace(0, 1, 11);
pc.classifier = 'svm';   % acc(SVM) > acc(KNN) for dyadic-edge(20,20,20)


%% setup & load data

switch(lower(pc.classifier))
  case 'svm'
    build_classifier = @(X,y) fitcecoc(X, y, 'Kfold', 5);
    
  case 'knn'
    build_classifier = @(X,y) fitcknn(X, y, 'NumNeighbors', 5, 'Standardize', 1, 'KFold', 5);
  otherwise
    error('unknown classifier');
end


load(sprintf('feats_%s.mat', pc.feature_type));
p % show parameters used to create data

%% Evaluate for different values of alpha (interpolation between max and avg)
X_max = feats.maxpool;
X_avg = feats.avgpool;
y = feats.y(:);

for ii = 1:length(pc.alpha_all)
    % features used are somewhere between max and avg
    alpha_i = pc.alpha_all(ii);
    Xi = alpha_i * X_max + (1 - alpha_i) * X_avg;

    tic
    model = build_classifier(Xi', y);
    y_hat = kfoldPredict(model);
    acc = sum(y(:) == y_hat(:)) / length(y_hat);
    
    fprintf('[%s]: took %0.2f seconds to fit and predict for %s:%0.2f\n', mfilename, toc, pc.feature_type, alpha_i);
    fprintf('[%s]: classification accuracy is %0.3f\n\n', mfilename, acc);

    % store results for later analysis
    if ii == 1
        Y_hat_all = zeros(numel(y_hat), length(pc.alpha_all));
        acc_all = zeros(length(pc.alpha_all),1);
    end
    Y_hat_all(:,ii) = y_hat;
    acc_all(ii) = acc;
end

save('classification_results.mat', 'Y_hat_all', 'acc_all', 'feats', 'pc', '-v7.3');



%% Some analysis of classification accuracy

y_avg = Y_hat_all(:,1);
y_max = Y_hat_all(:,end);


both_correct = (y_avg == y) & (y_avg == y_max);
only_avg_correct = (y_avg == y) & (y_avg ~= y_max);
only_max_correct = (y_max == y) & (y_avg ~= y_max);
neither_correct = (y_max ~= y) & (y_avg ~= y);

any_correct = any(bsxfun(@eq, Y_hat_all, y),2);
all_correct = sum(bsxfun(@eq, Y_hat_all, y),2) == size(Y_hat_all,2);
internal_correct = any_correct & (y_avg ~= y) & (y_max ~=y);

is_correct = bsxfun(@eq, Y_hat_all, y);

fprintf('[%s]: there are %d examples total\n', mfilename, numel(y));
fprintf('[%s]: any alpha correct:        %d\n', mfilename, sum(any_correct));
fprintf('[%s]: all alpha correct:        %d\n', mfilename, sum(all_correct));
fprintf('[%s]: both avg and max correct: %d\n', mfilename, sum(both_correct));
fprintf('[%s]: only alpha in (0,1):      %d\n', mfilename, sum(internal_correct));
fprintf('[%s]: only avg correct:         %d\n', mfilename, sum(only_avg_correct));
fprintf('[%s]: only max correct:         %d\n', mfilename, sum(only_max_correct));
fprintf('[%s]: neither correct:          %d\n', mfilename, sum(neither_correct));


% Take a look at which examples could be classified correctly if only we knew the correct alpha.
candidates = is_correct(any_correct & (~ all_correct), :);
[~,idx] = sort(sum(candidates,2));

figure; 
subplot(1,2,1);
imagesc(candidates(idx,:));
subplot(1,2,2);
histogram(sum(candidates,2));



%% Analysis

% it would be nice if there were monotonicity (or some easily
% distinguishable pattern) as we move between max and avg.

figure;
plot(pc.alpha_all, acc_all, '-o');
xlabel('alpha');
ylabel('accuracy');
  

[~,idx] = sort(sum(Y_hat_all,2));

figure;
imagesc(Y_hat_all(idx,:));
colorbar;

% See how often the estimate changes
is_unchanged = sum(bsxfun(@eq, Y_hat_all, y_avg),2) == size(Y_hat_all,2);

fprintf('[%s]: %d of %d estimates do not change as a function of alpha\n', mfilename, sum(is_unchanged), size(Y_hat_all,1));
