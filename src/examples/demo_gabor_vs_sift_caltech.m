

%% load data
caltech_dir = '../datasets/101_ObjectCategories';
sz = [200 200];

data = load_image_dataset(caltech_dir, sz);

nth = @(x,n) x(n);  % due to matlab slicing limitation
get_class_name = @(fn) nth(strsplit(fn, '/'), 4);

y_str = cellfun(get_class_name, data.files);


%% setup feature extractors
p_.sift.size = 4;
p_.sift.geom = [4 4 8];      % [nX nY nAngles]

sift_feats = @(I) dsift2(I, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

p_.gabor.M = size(data.X,1);
p_.gabor.b = p_.gabor.M / 12; 
p_.gabor.sigma = p_.gabor.b; 

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);

% limit # of gabor features to be the same as SIFT
G = G(:,:,1:prod(p_.sift.geom));

gabor_feats = @(I) abs(Gabor_transform(I, G));


%% Some manual analysis

% 20, 28 come from an experiment; 56, 60 from manual inspection
texture_dolphins = [20 28 56 60];  % local indices into the dolphin images
idx_dolphin = find(strcmp(y_str, 'dolphin'));

for di in texture_dolphins
    local_idx = idx_dolphin(id
    X = data.X(:,:,idx_dolphin(di));
    X_gabor = gabor_feats(X);
    X_sift = sift_feats(X);

    figure; imagesc(X); title(sprintf('dolphin %d', di));
    figure; imagesc(X_gabor(:,:,10)); title('gabor[10]');
    figure; imagesc(X_sift(:,:,10)); title('SIFT[10]');
end

