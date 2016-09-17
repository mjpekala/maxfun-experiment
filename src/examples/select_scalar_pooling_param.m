function pMax = select_scalar_pooling_param(X, y, f_pool, pAll, foldId)
% SELECT_SCALAR_POOLING_PARAM  Choose a pooling (hyper-)parameter 
%                              via cross-validation.
%
%  Exmaple:
%    pMax = select_scalar_pooling_param(X, y, f_pool, pAll [,foldIds]);
%
%   where,
%    X      : An tensor with dimensions (h, w, d, n) corresponding to
%             the image height, width, feature_dimension and
%             num_examples.
%    y      : An (n x 1) vector of class labels.
%    f_pool : A pooling function that takes two arguments: a subset of X
%             and a vector of scalar pooling parameters p.
%    pAll   : A set of scalar parameters to search over for the pooling function.
%
%  Uses mean accuracy obtained by applying a linear SVM to multiple
%  train/validation splits to decide which parameter is best.  
%
%  *** An implicit assumption is that the X provided by the caller
%  contains only training data ***

% mjp, april 2016

%% parameters and set up
verbose=true;

if nargin < 5
    foldId = assign_folds(y, 5);
end
allFoldIds = unique(foldId);
nFolds = length(allFoldIds);

assert(ndims(X) == 4);
assert(length(y) == size(X,4));
assert(length(y) == length(foldId));


if verbose,  vprintf = @fprintf;
else,  vprintf = @(varargin) 0; end


%% do it
vprintf('[%s]: Searching over %d parameter values using %d train/valid splits\n', ...
        mfilename, length(pAll), nFolds);

P = zeros(length(pAll), nFolds);
for ii = 1:nFolds
    vprintf('[%s]: starting split %d (of %d)\n', mfilename, ii, nFolds);

    % create a train/valid split
    isTrain = foldId ~= allFoldIds(ii);
    train.y = y(isTrain);
    valid.y = y(~isTrain);
    
    % apply pooling function 
    train.X = f_pool(X(:,:,:,isTrain), pAll);
    valid.X  = f_pool(X(:,:,:,~isTrain), pAll);
        
    for jj = 1:length(pAll)
        % evaluate performance
        % (the transpose is to put the #examples dimension first)
        [~,metrics] = eval_svm(train.X{jj}', train.y, valid.X{jj}', valid.y);
        P(jj,ii) = metrics.nCorrect;
    end
end


%% choose the best performing parameter (across all folds)

if 1
    metric = 'mean';
    perf = mean(P, 2);
else
    metric = 'sharpe';
    denom = max(1, var(P,0,2));
    perf = mean(P,2) ./ denom;
end

[~,idx] = max(perf);
pMax = pAll(idx);


vprintf('[%s]: Summary of results:\n', mfilename);
for ii = 1:length(perf)
    if ii == idx, annotation='(* best)';
    else, annotation=''; end
    
    vprintf('[%s]:   for p=%0.2g, %s=%0.2f %s\n', ...
            mfilename, pAll(ii), metric, perf(ii), annotation);
end
