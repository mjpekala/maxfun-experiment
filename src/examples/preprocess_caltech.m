% PREPROCESS CALTECH-101
%
% Caltech-101 does not have a native train/test split (as far as I
% am aware).  This script creates one.  It also:
%  - resizes the raw images to a consistent shape.
%  - discards any classes with insufficient representation

% mjp, october 2016


%% Setup / Parameters

imageDir = '../datasets/101_ObjectCategories';
sz = [200 200];    % note: our gabor feature code requires square images
seed = 9999;       % for repeatable results
n_test = 30;       % number of instances from each class to reserve
                   % for test.

n_train_min = 50;  % this needs to be large enough for validation


rng(seed);


%% load raw images
data = load_image_dataset(imageDir, sz);

if numel(data.y) == 0
    error('failed to load dataset!  Do the files exist?');
end

is_test = logical(zeros(size(data.y)));
is_train = logical(zeros(size(data.y)));
y_all = sort(unique(data.y));

fprintf('[%s]: withholding %d objects of each class for test\n', ... 
        mfilename, n_test);

for yi = unique(data.y(:)')
    idx = find(data.y==yi);
    if length(idx) < n_train_min + n_test
        fprintf('[%s]: Dropping class %d with %d instances\n', ...
                mfilename, yi, length(idx));
        continue;
    end
    
    idx_test = randsample(idx, n_test);
    idx_train = setdiff(idx, idx_test);
    is_train(idx_train) = true;
    is_test(idx_test) = true;
    clear idx idx_train idx_test;
end

train.y = data.y(is_train);
train.X = data.X(:,:,is_train);
train.files = data.files(is_train);
train.idx = find(is_train);
fprintf('[%s]: training data has %d classes and %d objects\n', ...
        mfilename, numel(unique(train.y)), numel(train.y));
save('caltech101_train.mat', 'train', '-v7.3');
clear train;

test.y = data.y(is_test);
test.X = data.X(:,:,is_test);
test.files = data.files(is_test);
test.idx = find(is_test);
fprintf('[%s]: test data has %d classes and %d objects\n', ...
        mfilename, numel(unique(test.y)), numel(test.y));
save('caltech101_test.mat', 'test', '-v7.3');
clear train;
