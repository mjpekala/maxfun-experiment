
% must process caltech data first
load('caltech_101_lean.mat');

[rows,cols,n_channels,n_images] = size(data.X);

fprintf('[%s]: X min/max : %0.2f / %0.2f\n', mfilename, min(data.X(:)), max(data.X(:)));


for ii = 1:100
    Xi = data.X(:,:,:,ii);
    p_max = max_pooling(Xi);
    p_avg = avg_pooling(Xi);
    [p_fun, w, loc] = maxfun_pooling(Xi, 20);
   
    % maxfun should sit between avg and max (at least when using
    % the default scaling)
    assert(all(floor(p_fun(:)) <= p_max(:)+eps));
    assert(all(p_avg(:) <= p_fun(:)+eps));
end


% visualize pooling region.
for channel = 1:n_channels
    figure;
    imagesc(data.X(:,:,channel,ii));
    colormap('bone'); colorbar;
    
    [r,c] = ind2sub([rows, cols], loc(channel));
    wc = w(channel)
    
    line([r, r+wc], [c,c], 'Color', 'g');
    line([r, r+wc], [c,c]+wc, 'Color', 'g');
    line([r, r], [c,c+wc], 'Color', 'g');
    line([r, r]+wc, [c,c+wc], 'Color', 'g');
end
