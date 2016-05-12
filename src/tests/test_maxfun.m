% Unit tests for maxfun pooling.
%

% mjp, april 2016

almost_equal = @(a,b) abs(a-b) <= 1e-6;

k = 3;
poolDim = 10;

avg_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'avg'), R, 'UniformOutput', 0));
max_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'max'), R, 'UniformOutput', 0));
sos_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'sos', k), R, 'UniformOutput', 0));
fun_pooling = @(R) cell2mat(cellfun(@(X) spatial_pool(X, 'fun', k), R, 'UniformOutput', 0));


%% Hand-crafted experiments

X = [ 2 2 0 0 ;
      2 2 0 0];
assert(spatial_pool(X, 'fun', 2) == 2);


X = [ 2 2 0 0 ;
      2 2 0 0 ;
      0 0 0 100];
assert(almost_equal(spatial_pool(X, 'fun', 2), 100/4));

X = [0 0 0 0;
     0 9 9 9;
     0 0 0 0;
     0 0 0 0];
assert(almost_equal(spatial_pool(X, 'fun', 2), 9*2/4));

X = [0 0 0 0;
     0 9 0 9;
     0 0 0 0;
     0 9 0 9];
assert(almost_equal(spatial_pool(X, 'fun', 2), 4));



X = zeros(100,100);  X(30:39, 30:39) = 1;
assert(almost_equal(spatial_pool(X, 'fun', 10), 1));
assert(spatial_pool(X, 'fun', 11) < 1);


%%  Test order relationship between MAXFUN and other techniques 

for ii = 1:10
    X = rand(100, 100); 
    R = pooling_regions(X, poolDim);

    Yavg = avg_pooling(R);
    Ymax = max_pooling(R);
    Ysos = sos_pooling(R);
    Yfun = fun_pooling(R);

    % Note the relationship between Yfun and Ysos assumes they both use
    % the same scalar parameter k (and the data is non-negative).
    % Otherwise, this is not true in general.
    assert(all(Yavg(:) <= Yfun(:)));
    assert(all(Yfun(:) <= Ysos(:)));
    assert(all(Ysos(:) <= Ymax(:)));
end


%% Exercise the multiple parameter feature

kVals = [2 4 6];
for ii = 1:10
    X = randn(100,100);
    y = spatial_pool(X, 'fun', kVals);  % whole image pooling
    
    assert(length(y) == length(kVals));
    
    for ii = 1:length(kVals)-1
        assert(y{ii} >= y{ii+1});
    end
end



%% also try a visual check
X = imread('peppers.png');
X = double(rgb2gray(X));
R = pooling_regions(X, poolDim);
Yfun = fun_pooling(R);
Yavg = avg_pooling(R);
Ymax = max_pooling(R);
Ysos = sos_pooling(R);

figure('Position', [100 100 1500 600]); 
subplot(2,5,1); imagesc(X, [0 255]);    title('orig');    
subplot(2,5,2); imagesc(Yavg, [0 255]); title('avg'); 
subplot(2,5,3); imagesc(Yfun, [0 255]); title('MAXFUN'); 
subplot(2,5,4); imagesc(Ysos, [0 255]); title('SOS');  
subplot(2,5,5); imagesc(Ymax, [0 255]); title('max');   

subplot(2,5,8); imagesc(Yfun-Yavg); title('MAXFUN - avg');  colorbar;
subplot(2,5,9); imagesc(Ysos-Yfun); title('SOS - MAXFUN');  colorbar;
subplot(2,5,10); imagesc(Ymax-Ysos); title('max - SOS');    colorbar;

colormap('gray');
