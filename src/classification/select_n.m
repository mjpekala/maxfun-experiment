function [selected, selectedM] = select_n(y, n, m)
% SELECT_N  Selects (up to) n objects from each class.
%
%     selected = select_n(y, n [, m])
%
%   where,
%      y        := an (n x 1) vector of class labels
%      n        := the number of examples from each class to choose
%      m        := a additional number of examples to choose from each class
%                  that is disjoint from the first n (optional).
%
%      selected := an (n x 1) logical vector where 1/true at index i
%                 means the ith object is included in the first
%                 data set.
%
%      selectedM := an (n x 1) logical vector where 1/true at index
%                   i means the ith object is included in the
%                   second data set (only applies if m > 0).
%
%  This function is really nothing more than a convenience wrapper
%  around randsample().  In a classification context, n would be
%  the number of examples to use for training while m would be the
%  number of test examples.  An alternative is to use assign_folds.


% mjp, april 2016

yAll = sort(unique(y));

if nargin < 3, m = 0; end;

% if no number of training examples is provided, use a ~50/50 train
% test split as the default.
if nargin < 2
    n = hist(y, length(yAll));
    assert(all(n > 0));
    n = floor(min(n)/2);
end

selected = logical(zeros(size(y)));
selectedM = logical(zeros(size(y)));

for yi = yAll(:)'
    idx = find(y == yi);  idx = idx(randperm(length(idx)));

    % choose n items
    ni = min(n, length(idx));
    selected(idx(1:ni)) = true;
   
    % (optional) choose m additional items
    mi = min(m, length(idx) - ni);
    if (mi > 0) && nargout > 1
        selectedM(idx(ni+1:ni+mi)) = true;
    end
end
