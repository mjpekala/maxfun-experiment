function rv = load_cifar_10(path, which_batch)
% LOAD_CIFAR_10  Loads one batch of CIFAR-10
%
%  Example usage:
%
%     data = load_cifar_10('./data/cifar-10-batches-mat', 1);
%     imagesc(data(:,:,:,1));
%     


if which_batch > 0
    filename = sprintf('data_batch_%d.mat', which_batch);
else
    filename = 'test_batch.mat';
end

filename = fullfile(path, filename);

load(filename);

rv.X = reshape(data, size(data,1), 32, 32, 3);
rv.X = permute(rv.X, [3,2,4,1]);
rv.y = labels;

clear data labels;


