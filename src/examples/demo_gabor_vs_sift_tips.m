

%% load data
data_dir = '../datasets/KTH_TIPS';
sz = [200 200];

data = load_image_dataset(data_dir, sz);


%% setup feature extractors
p_.sift.size = 4;
p_.sift.geom = [4 4 8];      % [nX nY nAngles]

sift_xform = @(I) dsift2(I, 'step', 1, ...
                         'size', p_.sift.size, ...
                         'geometry', p_.sift.geom);

p_.gabor.M = size(data.X,1);
p_.gabor.b = p_.gabor.M / 12; 
p_.gabor.sigma = p_.gabor.b; 

G = Gabor_construct(p_.gabor.M, p_.gabor.b, p_.gabor.sigma);

% limit # of gabor features to be the same as SIFT
G = G(:,:,1:prod(p_.sift.geom));

gabor_xform = @(I) abs(Gabor_transform(I, G));


%% Some manual analysis

X_foil = data.X(:,:,data.y==1);


%%

for yi = unique(data.y(:)')
    Xi = data.X(:,:,find(data.y == yi, 1));
    figure; imagesc(Xi);  colormap('gray');  
    title(data.class_names{yi});
    figure; view_feats(gabor_xform(Xi), 5); 
    title(sprintf('%s gabor', data.class_names{yi}));
    figure; view_feats(sift_xform(Xi), 5); 
    title(sprintf('%s sift', data.class_names{yi}));
end


%% 
e_gabor = squeeze(sum(sum(sum(X_foil_gabor,1), 2), 4));

figure; stem(e_gabor);
xlabel('feature dimension');
ylabel('sum');
