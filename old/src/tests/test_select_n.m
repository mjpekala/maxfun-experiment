% mjp, april 2016

n = 5;
y = [ones(2*n,1) ; 2*ones(2*n,1) ; 3*ones(2*n,1)];

isTrain = select_n(y,n);
yTrain = y(isTrain);
yTest = y(~isTrain);

assert(sum(yTrain==1) == n);
assert(sum(yTrain==2) == n);
assert(sum(yTrain==3) == n);
assert(sum(yTest==1) == n);
assert(sum(yTest==2) == n);
assert(sum(yTest==3) == n);


% make sure you get a reasonable result if you ask for too many
% items.
y = ones(10,1);
bits = select_n(y, length(y)+1);
assert(sum(bits) == length(y));


% Try out the three argument form.
n = 5;
y = [ones(2*n,1) ; 2*ones(2*n,1) ; 3*ones(2*n,1)];

[foo,bar] = select_n(y, n, n);
for ii = 1:3
    assert(sum(y(foo)==ii) == sum(y(bar)==ii));
end


[foo,bar] = select_n(y, n+1, n);
for ii = 1:3
    assert(sum(y(foo)==ii) == n+1);
    assert(sum(y(bar)==ii) == n-1);
end

