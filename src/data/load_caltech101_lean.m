function lean_data = load_caltech101_lean(image_dir, dim)
% LOAD_CALTECH_101_LEAN  Load a subset Caltech-101. 
%
%   Here we return classes with non-trivial but comparable number
%   of instances.


n_examp_min = 80;
n_examp_max = 120;

data = load_caltech101(image_dir, dim);
y_all = unique(data.y);


y_remap = [];
keepme = logical(zeros(size(data.y)));

for yi = y_all
    n_instances = sum(data.y == yi);
    
    if n_examp_min <= n_instances  && n_instances < n_examp_max
        y_remap = [y_remap yi];
        keepme(data.y == yi) = true;
        fprintf('[%s]: keeping class %d (%s) with %d instances\n', mfilename, yi, data.class_names{yi}, n_instances);
    end
end


lean_data.X = data.X(:,:,:,keepme);
lean_data.y = data.y(keepme);
lean_data.files = data.files(keepme);
lean_data.was_grayscale = data.was_grayscale(keepme);
lean_data.y_remap = y_remap;


% map from original class labels to new labels
for ii = 1:length(y_remap)
    bits = lean_data.y == y_remap(ii);
    lean_data.y(bits) = ii;
end

