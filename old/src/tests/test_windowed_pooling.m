% TEST_WINDOWED_POOLING

f_pool = @(I) spatial_pool(I, 'avg');


%% 2d case
X = randn(100,100);
X(1:20,1:20) = 17;

Xp = windowed_pooling(X, 20, f_pool);

assert(size(Xp,1) == 5);
assert(size(Xp,2) == 5);
assert(Xp(1,1) == 17);
assert(Xp(1,2) ~= 17);


%% 3d case
X3d = cat(3, X, X, X);

Xp = windowed_pooling(X3d, 20, f_pool);

assert(size(Xp,1) == 5);
assert(size(Xp,2) == 5);
assert(size(Xp,3) == 3);
assert(all(Xp(1,1,:) == 17));
assert(all(Xp(1,2,:) ~= 17));
