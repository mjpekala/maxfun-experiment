
% mjp, april 2016

% Make sure all_windowed_sums does the same thing as the supposedly
% equivalent convolution.

X = rand(300,300);
wAll = [5:10 20:30];
Z = all_windowed_sums(X, wAll);
assert(size(Z,3) == numel(wAll));

Z2 = zeros(size(X,1), size(X,2), numel(wAll));
for ii=1:length(wAll)
    w = wAll(ii);
    Zi = conv2(X, ones(w,w), 'valid') / w / w;
    Z2(1:size(Zi,1), 1:size(Zi,2), ii) = Zi;
end

delta = max(abs(Z(:) - Z2(:)));
assert(delta < 1e-10);


