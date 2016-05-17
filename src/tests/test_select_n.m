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
