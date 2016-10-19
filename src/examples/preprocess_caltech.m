% PREPROCESS CALTECH-101
%
% Caltech-101 does not have a native train/test split (as far as I
% am aware).  This script creates one.  It also resizes the images
% to a consistent shape.

% mjp, october 2016



%% load raw images
imageDir = '../datasets/101_ObjectCategories';
sz = [200 200];    % note: our gabor feature code requires square images
seed = 9999;       % for repeatable results
n_test = 30;       % number of instances from each class to reserve
                   % for test

rng(seed);

data = load_image_dataset(imageDir, sz);

if numel(data.y) == 0
    error('failed to load dataset!  Do the files exist?');
end

is_test = logical(zeros(size(data.y)));
y_all = sort(unique(data.y));

fprintf('[%s]: withholding %d objects of each class for test\n', ... 
        mfilename, n_test);

for yi = unique(data.y(:)')
    idx = randsample(find(data.y==yi), n_test);
    is_test(idx) = true;
end

train.y = data.y(~is_test);
train.X = data.X(:,:,~is_test);
train.files = data.files(~is_test);
train.idx = find(~is_test);
fprintf('[%s]: training data has %d objects\n', mfilename, numel(train.y));
save('caltech101_train.mat', 'train', '-v7.3');
clear train;

test.y = data.y(is_test);
test.X = data.X(:,:,is_test);
test.files = data.files(is_test);
test.idx = find(is_test);
fprintf('[%s]: test data has %d objects\n', mfilename, numel(test.y));
save('caltech101_test.mat', 'test', '-v7.3');
clear train;
