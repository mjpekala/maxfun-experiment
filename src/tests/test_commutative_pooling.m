% Unit tests for pooling operations.

% mjp, april 2016

almost_equal = @(a,b) abs(a-b) <= 1e-6;


%% Make sure average and max pooling behave as expected
X = randn(256, 256, 3);

xAvg = spatial_pool(X, 'avg');
assert(numel(xAvg) == size(X,3));
assert(almost_equal(xAvg(1), mean(mean(X(:,:,1)))));
assert(almost_equal(xAvg(end), mean(mean(X(:,:,end)))));

xMax = spatial_pool(X, 'max');
assert(numel(xMax) == size(X,3));
assert(almost_equal(xMax(1), max(max(X(:,:,1)))));
assert(almost_equal(xMax(end), max(max(X(:,:,end)))));

X = randn(256, 256, 3, 20);
xAvg = spatial_pool(X, 'avg');
assert(all(size(xAvg) == [size(X,3), size(X,4)]));
assert(almost_equal(xAvg(1,1), mean(mean(X(:,:,1,1)))));


%% Make sure SOS pooling behaves as expected

X = rand(30,30,10);
k = 3;
Y = spatial_pool(X, 'sos', k);

for ii = 1:size(X,3)
    v = X(:,:,ii);
    v = sort(v(:), 'descend');
    v = sum(v(1:k)) / k;
    assert(almost_equal(Y(ii), v));
end

% Same test, but for 4-d tensors (i.e. matrices). 
% 
X = rand(10,10,3,3);
Y = spatial_pool(X, 'sos', k);

for ii = 1:size(X,3)
    for jj = 1:size(X,4)
        v = X(:,:,ii,jj);
        v = sort(v(:), 'descend');
        v = sum(v(1:k)) / k;
        assert(almost_equal(Y(ii,jj), v));
    end
end


% just make sure this is working as expected
x = zeros(3,1);
x(1:3) = 3;
assert(spatial_pool(x, 'sos', 3) == 3);


%% Make sure p-norm pooling behaves as expected
for ii = 1:10
    x = randn(100,1);
    xp = norm(x, 2);
    xp2 = spatial_pool(x, 'pnorm', 2) * (length(x).^(1/2));
    assert(almost_equal(xp, xp2));
end


%% Test out the ability to evalute multiple pooling parameters
X = rand(10,10,3,3);
kVals = 3:5;
Y = spatial_pool(X, 'sos', kVals);

assert(length(Y) == length(kVals));
for ii = 1:length(Y)
    Yi = Y{ii};
    Y2 = spatial_pool(X, 'sos', kVals(ii));
    assert(all(Yi(:) == Y2(:)));
end


%% Natural image example
avg_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'avg'), R, 'UniformOutput', 0));
max_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'max'), R, 'UniformOutput', 0));
sos_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'sos', 10), R, 'UniformOutput', 0));
pnorm_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'pnorm', 2), R, 'UniformOutput', 0));

X = imread('peppers.png');
X = double(rgb2gray(X));
R = pooling_regions(X, 10);
Yavg = avg_pooling(R);
Ymax = max_pooling(R);
Ysos = sos_pooling(R);
Ypn2 = pnorm_pooling(R);

% there is an order relationship
assert(all(Yavg(:) <= Ysos(:)));
assert(all(Ysos(:) <= Ymax(:)));

% visualize (requires tight_subplot.m)
if exist('tight_subplot')
    showit = @(X) set(get(imagesc(X, [0, 255]), 'Parent'), 'XTick', [], 'YTick', []);
    showit_delta = @(X) set(get(imagesc(X, [-20, 20]), 'Parent'), 'XTick', [], 'YTick', []);

    figure('Position', [100 100 1200 1200]); 
    ha = tight_subplot(5, 5, [.025 .025], [.01 .05], .01);
    
    axes(ha(1)); showit(X);  title('orig');    
    axes(ha(2)); showit(Yavg);  title('avg'); 
    axes(ha(3)); showit(Ysos);  title('sos(10)');
    axes(ha(4)); showit(Ymax);  title('max'); 
    axes(ha(5)); showit(Ypn2);  title('sqrt'); 
    
    axes(ha(6)); showit(Yavg);  ylabel('avg'); 
    axes(ha(7)); hist(Yavg(:), 30);
    axes(ha(8)); showit_delta(Yavg - Ysos);
    axes(ha(9)); showit_delta(Yavg - Ymax);
    axes(ha(10)); showit_delta(Yavg - Ypn2);
    
    axes(ha(11)); showit(Ysos);  ylabel('sos(10)'); 
    axes(ha(12)); showit_delta(Ysos - Yavg);  
    axes(ha(13)); hist(Ysos(:), 30);
    axes(ha(14)); showit_delta(Ysos - Ymax);
    axes(ha(15)); showit_delta(Ysos - Ypn2);
    
    axes(ha(16)); showit(Ysos);  ylabel('max'); 
    axes(ha(17)); showit_delta(Ymax - Yavg);  
    axes(ha(18)); showit_delta(Ymax - Ysos);
    axes(ha(19)); hist(Ymax(:), 30);
    axes(ha(20)); showit_delta(Ymax - Ypn2);
    
    axes(ha(21)); showit(Ypn2);  ylabel('sqrt'); 
    axes(ha(22)); showit_delta(Ypn2 - Yavg);  
    axes(ha(23)); showit_delta(Ypn2 - Ysos);
    axes(ha(24)); showit_delta(Ypn2 - Ymax);
    axes(ha(25)); hist(Ypn2(:), 30);
else
    fprintf('[%s]: tight_subplot not found in matlab search path; omitting figure\n', mfilename);
end

