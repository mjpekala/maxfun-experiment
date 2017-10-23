

data = load_cifar_10('./cifar-10-batches-mat', 1);
xi = data.X(:,:,:,1);

% TEST: window size is entire image
xi_hat = extract_all_windows(xi, [32, 32], 32);
assert(all(xi(:) == xi_hat(:)));

% TEST: extracting disjoint windows
xi_hat = extract_all_windows(xi, [16, 16], 16);
assert(size(xi_hat,3) == 4*3);

% TEST: overlapping windows
xi_hat = extract_all_windows(xi, [16, 16], 8);
assert(size(xi_hat,3) == 9*3);
