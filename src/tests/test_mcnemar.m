% TEST_MCNEMAR
%
%  Compare hand-rolled implementation of McNemar's test
%  with matlab's built-in.

Yhat1 = [1 1 1 1 1 0 0 0 0 1 ;
         1 1 1 1 1 0 0 0 0 0 ;
         1 1 1 1 0 0 0 1 0 1 ];

Yhat2 = [1 1 1 0 1 1 1 0 0 0 ;
         1 1 1 1 1 0 0 0 0 1 ;
         1 1 1 1 1 0 0 0 0 0];

y =  [1 1 1 1 1 0 0 0 0 0];

for ii = 1:size(Yhat1,1)
    p = mcnemar(Yhat1(ii,:), Yhat2(ii,:), y);
    [h,p2] = testcholdout(Yhat1(ii,:), Yhat2(ii,:), y);
    assert(abs(p - p2) < 1e-7);
end



% monte carlo
for ii = 1:100
    yhat1 = double([rand(5,1) < .7  ; rand(5,1) < .2]);
    yhat2 = double([rand(5,1) < .75 ; rand(5,1) < .25]);
    
    p = mcnemar(yhat1, yhat2, y);
    [h,p2,e1,e2] = testcholdout(yhat1, yhat2, y(:));
    assert(abs(p - p2) < 1e-7);
end
