function X = make_composite_image(X_texture, dim, n_edges, color_range)
% MAKE_COMPOSITE_IMAGE 
%  Creates an image consisting of a texture patch and some lines.

assert(all(dim > size(X_texture)));

[r,c] = size(X_texture);

X = zeros(dim,dim);

% translate the texture into the overall image
tr = 1 + floor((dim - r) * rand);
tc = 1 + floor((dim - c) * rand);
X(tr:tr+r-1, tc:tc+c-1) = X_texture;

% draw some clutter
p0 = floor(1 + rand(n_edges,2) * dim);
p1 = floor(1 + rand(n_edges,2) * dim);

color = color_range(1) + rand()*(color_range(2) - color_range(1));

X = draw_lines(X, p0, p1, color);


