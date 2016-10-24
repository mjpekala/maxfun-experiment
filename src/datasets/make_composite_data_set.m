% MAKE_COMPOSITE_DATA_SET
% 
%  Creates a synthetic data set consisting of textures patches 
%  (from KTH-TIPS) together with some clutter.


rng(9999, 'twister');

%p_.patch_sz = 50;
%p_.image_sz = 100;
p_.patch_sz = 200;
p_.image_sz = 300;
p_.n_per_class = 100;
p_.n_clutter_lines = 0;  % TODO
p_.color_range = [100 200];

make_dir = @(dirName) ~exist(dirName) && mkdir(dirName);


%% load texture data
data = load_image_dataset('../datasets/KTH_TIPS', p_.patch_sz*[1 1]);
data.X = single(data.X);


%% Create composite data
root_dir = 'UMD_Composite';
make_dir(root_dir);

for yi = 1:length(data.class_names)
    subdir = fullfile(root_dir, data.class_names{yi});
    make_dir(subdir);
   
    % sample textures (with replacement)
    y_idx = find(data.y == yi);
    textures = randsample(y_idx, p_.n_per_class, true);

    for ii = 1:p_.n_per_class
        Xi = make_composite_image(data.X(:,:,textures(ii)), ...
                                  p_.image_sz, ...
                                  p_.n_clutter_lines, ...
                                  p_.color_range);
        fn = sprintf('image_%03d.png', ii);
        imwrite(Xi/255, fullfile(subdir, fn), 'png');
    end
end

