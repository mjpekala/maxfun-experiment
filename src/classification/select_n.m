function selected = select_n(y, n)
% SELECT_N  Selects n objects from each class.
%
%     selected = select_n(y, n)
%
%   where,
%      y        := an (n x 1) vector of class labels
%      n        := the number of examples from each class to choose
%
%      selected := an (n x 1) logical vector where 1/true at index i
%                 means the ith object is included in the training
%                 data set.
%
%  This function is really nothing more than a convenience wrapper
%  around randsample().

% mjp, april 2016

yAll = sort(unique(y));

% if no number of training examples is provided, use a ~50/50 train
% test split as the default.
if nargin < 2
    n = hist(y, length(yAll));
    assert(all(n > 0));
    n = floor(min(n)/2);
end

selected = logical(zeros(size(y)));
for yi = yAll(:)'
    idx = find(y == yi);
    idx = randsample(idx, n);
    selected(idx) = true;
end
