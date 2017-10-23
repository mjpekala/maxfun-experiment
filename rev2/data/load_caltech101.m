function dataset = load_caltech101(image_dir, dim, extension)
%  LOAD_CALTECH101    Loads Caltech 101 dataset
%
if nargin < 2, dim = 128; end
if nargin < 3, extension='.jpg'; end
verbose = true;


% Get a list of all classes in the dataset.
% e.g. for Caltech 101, there will be 101 subdirectories.
class_dirs = dir(image_dir);

% Omit the '.', '..' and any other special directories.
%
% In the case of caltech 101, also exclude the background category
% (not technically one of the 101 classes).
%
% Also exclude any non-directory files (e.g. readme files) that may
% live in the top level directory.
%
included = logical(ones(length(class_dirs),1));
for ii = 1:length(class_dirs)
    if strcmp(class_dirs(ii).name(1), '.')
        included(ii) = 0;
    elseif strcmp(class_dirs(ii).name, 'BACKGROUND_Google')
        included(ii) = 0;
    elseif ~isdir(fullfile(image_dir, class_dirs(ii).name))
        included(ii) = 0;
    end
end
class_dirs = class_dirs(included);
clear included;


% create a list of all class names
class_names = {};
for ii = 1:length(class_dirs)
    class_names{ii} = class_dirs(ii).name;
end

% impose a canonical ordering (in case we don't want to rely on OS order)
[class_names,idx] = sort(lower(class_names));  % impose canonical ordering
class_dirs = class_dirs(idx);


%% Load images
y = [];
for yi = 1:length(class_names)
    % get a list of all images in this subdirectory
    class_dir = fullfile(image_dir, class_dirs(yi).name);
    path = fullfile(class_dir, ['*' extension]);
    files = dir(path);
    
    if isempty(files)
        warning(sprintf('directory %s has no images with extension %s', path, extension));
        continue;
    end
   
    % convert local filename into a full path
    files = cellfun(@(yi) fullfile(class_dir, files(yi).name), ...
                    num2cell(1:length(files)), 'UniformOutput', 0);

    if yi == 1
        all_files = files;
    else
        all_files = [all_files files];
    end
    
    y = [y   yi*ones(1, length(files))];
end


fprintf('[%s]: reading images...please wait...\n', mfilename);

X = zeros(dim, dim, 3, numel(y), 'uint8');
was_grayscale = logical(size(y));

for ii = 1:length(y)
    xi = imread(all_files{ii});
    
    % some images are grayscale, evidently
    % in these cases, create a synthetic 3 channel image.
    if size(xi,3) == 1
        xi = cat(3, xi, xi, xi);
        was_grayscale(ii) = true;
    end
    
    X(:,:,:,ii) = resize_square(xi, dim);
end


if verbose
    figure;
    h = histogram(y, 101);
    title('Caltech-101 class membership');
    xlabel('class');  ylabel('num instances');
    
    figure;
    plot(sort(h.Values, 'descend'), '-o');
    grid on;
    title('Caltech-101 class membership');
    xlabel('rank');  ylabel('num instances');
end


dataset.X = X;
dataset.y = y;
dataset.files = all_files;
dataset.was_grayscale = was_grayscale;
dataset.class_names = class_names;
