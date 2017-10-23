function view_feats(X, n)

if nargin < 2, n = 3; end

Img = cat3(X(:,:,1:n));

for ii = 2:n
    a = (ii-1)*n+1;
    b = a + n - 1;
    Img = [Img ; cat3(X(:,:,a:b))];
end

imagesc(Img);
colorbar;
colormap('gray');
set(gca, 'YTick', [], 'XTick', []);



function X_out = cat3(X)
% CAT3  Concatenate a tensor along dimension #3.

[m,n,p] = size(X);

X_out = zeros(m, n*p);

for ii = 1:p
    c = 1 + (ii-1) * n;
    X_out(:, c:c+n-1) = X(:,:,ii);
end
