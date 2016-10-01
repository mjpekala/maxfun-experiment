% unit tests for pooling_regions.m

% mjp, april 2016


%% test matrix inputs
X = rand(300,300);

R = pooling_regions(X, 7);
Z = cell2mat(R);
Z = Z(1:size(X,1), 1:size(X,2));
assert(all(X(:) == Z(:)));

R11 = R{1,1};
X11 = X(1:7, 1:7);
assert(all(R11(:) == X11(:)));


%% test tensor inputs

X = zeros(300,300,5);   % 5 feature maps
for ii = 1:size(X,3), X(:,:,ii) = ii; end

Y = pooling_regions(X, 100);

assert(size(Y,1) == 3);
assert(size(Y,2) == 3);
assert(size(Y,3) == 1);

for ii = 1:size(Y,1)
    for jj = 1:size(Y,2)
        Yij = Y{ii,jj};
        assert(size(Yij,1) == 100);
        assert(size(Yij,2) == 100);
        assert(size(Yij,3) == 5);
        for kk = 1:size(Yij,3)
            assert(all(all(Yij(:,:,kk) == kk)));
        end
    end
end


% note to self: can stack regions along the feature dimension
Tmp = cat(3, Y{:});
assert(all(all(Tmp(:,:,1) == Y{1,1}(:,:,1))));
assert(all(all(Tmp(:,:,end) == Y{end,end}(:,:,end))));


%%
fprintf('[%s]: all tests passed!\n', mfilename);