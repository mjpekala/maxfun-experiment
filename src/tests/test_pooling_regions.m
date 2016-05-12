% unit tests for pooling_regions.m

% mjp, april 2016

% This function is pretty simple, so there's not that much that is
% likely to go wrong.  Hence, these are just some basic sanity checks.

X = rand(300,300);

R = pooling_regions(X, 7);
Z = cell2mat(R);
Z = Z(1:size(X,1), 1:size(X,2));
assert(all(X(:) == Z(:)));

R11 = R{1,1};
X11 = X(1:7, 1:7);
assert(all(R11(:) == X11(:)));

%------------------------------
fprintf('[%s]: all tests passed!\n', mfilename);