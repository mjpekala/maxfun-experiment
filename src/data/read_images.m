function X = read_images(dirName, sz)
% READ_IMAGES  Reads image data for classification problems.
%
%     X = read_images(dirName, sz)
%           - OR -
%     X = read_images(fileName, sz)
%
%  where,
%
%     dirName := a directory containing one class of image data
%                       - OR -
%                a single image file name
%
%     sz := the (height,width) to make each image (via resizing)
%
%     X := a cell array contining each image.  This is a cell (as
%                opposed to a tensor) in case the caller did not elect
%                to resize the images.  If the caller only provided a
%                single file name, then instead of a cell array X is
%                just the image.

% mjp, april 2016

if nargin < 2, sz = []; end

if isdir(dirName)
    % laod all images in directory 
    files = dir(fullfile(dirName, '*.jpg'));
    if isempty(files), files = dir(fullfile(dirName, '*.png')); end
        
    if isempty(files)
        error(sprintf('"%s" does not contain known image types', dirName));
    end

    X = cellfun(@(ii) load_one_file(fullfile(dirName, files(ii).name), sz), ...
                num2cell(1:length(files)), 'UniformOutput', 0);
else
    X = load_one_file(dirName, sz);
end


function I = load_one_file(fn, sz)
% LOAD_ONE_FILE  Loads a single image file

I = imread(fn);

% convert to grayscale (if not already)
if ndims(I) == 3
    I = single(rgb2gray(I));
else
    I = single(I);
end
    
% (optional) resize image
if ~isempty(sz)
    I = imresize(I, sz);
end
