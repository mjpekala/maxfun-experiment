function isTrain = select_training(y, nTrain)
% SELECT_TRAINING  Identifies training data.
%
%     isTrain = select_training(y, nTrain)
%
%   where,
%      y       := an (n x 1) vector of class labels for the entire 
%                 data set
%      nTrain  := the number of examples from each class to use for
%                 training
%
%      isTrain := an (n x 1) logical vector where 1/true at index i
%                 means the ith object is included in the training
%                 data set.
%
%  Chooses nTrain objects from each class uniformly at random.
%  Returns a logical vector indicating which objects are in the
%  training set.

% mjp, april 2016

yAll = sort(unique(y));

% if no number of training examples is provided, use a ~50/50 train
% test split as the default.
if nargin < 2
    n = hist(y, length(yAll));
    assert(all(n > 0));
    nTrain = floor(min(n)/2);
end

isTrain = logical(zeros(size(y)));

for yi = yAll(:)'
    idx = find(y == yi);
    idx = randsample(idx, nTrain);
    isTrain(idx) = true;
end
