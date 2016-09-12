function Y = map_image(T, f, verbose)
% MAP  Applies function to a set of images.
%
%    T : A tensor of images with dimensions (m x n x z) which
%        represents a stack of (m x n) images (n in total).
%
%    f : A function that accepts a single input argument, an (m x n)
%        image.
%
%    Y : A tensor of processed images, where the last dimension is
%        z.  The first n-1 dimensions depend upon f.
%
% 

% mjp, may 2016

if nargin < 3, verbose=true; end

z = size(T,3);
if z <= 0, error('input is not a 3d tensor'); end

timer = tic;  lastChatter = -1e3;


for ii=1:z
    y = f(T(:,:,ii));
    if ii == 1
        Y = zeros(numel(y), z);
    end
    Y(:,ii) = y(:);

    if verbose && ((toc(timer) - lastChatter) > 20)
        fprintf('[%s]: processed %d items (of %d) in %0.2f minutes\n', ...
                mfilename, ii, z, toc(timer)/60);
        lastChatter = toc(timer);
    end
end

Y = reshape(Y, [size(y) z]);

if verbose 
    fprintf('[%s]: processed %d items (of %d) in %0.2f minutes\n', ...
            mfilename, ii, z, toc(timer)/60);
end

