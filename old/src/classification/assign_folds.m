function foldId = assign_folds(y, nFolds)
% ASSIGN_FOLDS  Assigns objects to one of n folds.
%
%       fid = assign_folds(y, n);
%
%    where,
%      y := an (m x 1) vector of class labels.
%      n := the number of folds (scalar) 
%
%  An attempt is made to ensure each fold reflects the same relative
%  class distribution as y.

% mjp, april 2016

foldId = zeros(size(y));

yAll = unique(y);
for ii = 1:length(yAll), yi = yAll(ii);
    idx = find(y == yi);
    idx = idx(randperm(length(idx)));  % shuffle
    ni = length(idx);
    
    fid = repmat(1:nFolds, 1, ceil(ni/nFolds));
    fid = fid(1:ni);
    foldId(idx) = fid;
end

assert(all(foldId > 0));
