function X = make_composite_image(X_texture, dim, n_edges, color_range)
% MAKE_COMPOSITE_IMAGE 
%  Creates an image consisting of a texture patch and some lines.

assert(all(dim > size(X_texture)));

[r,c] = size(X_texture);

if 1
    X = zeros(dim,dim);
else
    % background = gaussian noise
    X = 80*rand(dim,dim);
end



% transform texture (optional)
if 1
    if rand() < .5, X_texture = fliplr(X_texture); end
    if rand() < .5, X_texture = flipud(X_texture); end
end


% add clutter edges
if n_edges > 0
    p0 = floor(1 + rand(n_edges,2) * dim);
    p1 = floor(1 + rand(n_edges,2) * dim);

    color = color_range(1) + rand()*(color_range(2) - color_range(1));

    X = draw_lines(X, p0, p1, color);
end


% translate the texture into the overall image
tr = 1 + floor((dim - r) * rand);
tc = 1 + floor((dim - c) * rand);
X(tr:tr+r-1, tc:tc+c-1) = X_texture;
