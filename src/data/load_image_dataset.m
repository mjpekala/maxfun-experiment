function data = load_image_dataset(imageDir, sz)
%  LOAD_IMAGE_DATASET    Loads images from file.
%
%   Loads an image data set; an assumption is that objects from each
%   class have been placed in separate subdirectories (e.g. as per the
%   default layout for Caltech 101).
%
%   Note there there is a bit of dataset-specific logic below
%   (for Caltech-101 in particular).
%
%   Example:
%      data = load_image_dataset('~/Data/caltech_101/101_ObjectCategories', [200 300]);
%      figure; imagesc(data.X(:,:,1));
%
%   where 
%      imageDir  : A string containing the top level directory
%                  containing all the image subdirectories. 
%
%      sz        : A pair [height, width] indicating how the
%                  raw images should be resized.  If sz is
%                  empty, the data will not be loaded (can be
%                  useful if it will to too memory intensive to
%                  load the entire data set at once).
%

% mjp, april 2016



% Get a list of all classes in the dataset.
% e.g. for Caltech 101, there will be 101 subdirectories.
classDirs = dir(imageDir);

% Omit the '.', '..' and any other special directories.
%
% In the case of caltech 101, also exclude the background category
% (not technically one of the 101 classes).
%
% Also exclude any non-directory files (e.g. readme files) that may
% live in the top level directory.
%
included = logical(ones(length(classDirs),1));
for ii = 1:length(classDirs)
    if strcmp(classDirs(ii).name(1), '.')
        included(ii) = 0;
    elseif strcmp(classDirs(ii).name, 'BACKGROUND_Google')
        included(ii) = 0;
    elseif ~isdir(fullfile(imageDir, classDirs(ii).name))
        included(ii) = 0;
    end
end
classDirs = classDirs(included);


% create a list of all class names
classNames = {};
for ii = 1:length(classDirs)
    classNames{ii} = classDirs(ii).name;
end
data.class_names = classNames;


% get a list of all individual image filenames.
yAll = 1:length(classNames);
data.y = [];
data.files = {};
for yi = yAll
    cclass = classDirs(yi).name;
    % files can be either .jpg or .png (but not a mixture of both)
    files = dir(fullfile(imageDir, cclass, '*.jpg'));
    if isempty(files), files = dir(fullfile(imageDir, cclass, '*.png')); end
    assert(~isempty(files));
    
    files = cellfun(@(ii) fullfile(imageDir, cclass, files(ii).name), ...
                    num2cell(1:length(files)), 'UniformOutput', 0);
    
    data.files = [data.files files];
    data.y = [data.y ; yi*ones(length(files),1)];
end

fprintf('[%s]: Data set has %d objects and %d classes\n', mfilename, length(data.y), length(yAll));

if ~isempty(sz)
    fprintf('[%s]: reading images...please wait...\n', mfilename);
    data.X = cellfun(@(fn) read_images(fn, sz), data.files, 'UniformOutput', 0);
    data.X = cat(3, data.X{:});
end
