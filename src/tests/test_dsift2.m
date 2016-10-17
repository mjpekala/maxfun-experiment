

I = single(imread('Lena.jpg'))/255;
[m,n] = size(I);

figure; imagesc(I);

% try SIFT with default parameters
X1 = dsift2(I);
assert(size(X1,3) == 128);

% try DSIFT with only 64 dimensions
geom = [4 4 4];
X2 = dsift2(I, 'geometry', geom);
assert(size(X2,3) == 4*4*4);
