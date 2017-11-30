% unit tests for pooling_regions.m

% mjp, april 2016


as_column = @(x) x(:);


%% test matrix inputs
X = rand(300,300);

p = 7;
R = pooling_regions(X, p);
Z = cell2mat(R);
Z = Z(1:size(X,1), 1:size(X,2));
assert(all(X(:) == Z(:)));

X11 = X(1:p, 1:p);
assert(all(R{1,1}(:) == as_column(X(1:p, 1:p))));

% try the alternate return value
Rt = pooling_regions(X, p, true);
assert(ndims(Rt) == 3);
assert(size(Rt,1) == p);
assert(size(Rt,2) == p);
assert(size(Rt,3) == size(R,1)*size(R,2));


%% test tensor inputs

X = zeros(300,300,5);   % 5 feature maps
for ii = 1:size(X,3), X(:,:,ii) = ii; end

p = 100;
Y = pooling_regions(X, p);

assert(size(Y,1) == 3);
assert(size(Y,2) == 3);
assert(size(Y,3) == 1);

for ii = 1:size(Y,1)
    for jj = 1:size(Y,2)
        Yij = Y{ii,jj};
        assert(size(Yij,1) == p);
        assert(size(Yij,2) == p);
        assert(size(Yij,3) == 5);
        for kk = 1:size(Yij,3)
            assert(all(all(Yij(:,:,kk) == kk)));
        end
    end
end


% try the alternate return value
Yt = pooling_regions(X, p, true);
assert(ndims(Yt) == 3);
assert(size(Yt,1) == p);
assert(size(Yt,2) == p);
assert(size(Yt,3) == size(Y,1)*size(Y,2)*5);


%%
fprintf('[%s]: all tests passed!\n', mfilename);