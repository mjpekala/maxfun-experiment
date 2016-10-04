% TEST_MCNEMAR
%

% mjp, october 2016

%%  Compare with matlab's built in test (if available)

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
    yhat1 = double([rand(50,1) < .7  ; rand(50,1) < .2]);
    yhat2 = double([rand(50,1) < .75 ; rand(50,1) < .25]);
    y = [ones(50,1) ; zeros(50,1)];
    
    p = mcnemar(yhat1, yhat2, y);
    [h,p2,e1,e2] = testcholdout(yhat1, yhat2, y(:));
    assert(abs(p - p2) < 1e-7);
end
