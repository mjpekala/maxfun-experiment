function [p, e_10, e_01] = mcnemar(y_hat_1, y_hat_2, y)
%  MCNEMAR Runs McNemar's test for paired data [1]
%
%
%  H_0 = null hypothesis:      \mu_0 =  \mu_1
%  H_1 = alternate hypothesis: \mu_0 ~= \mu_1
%
%  e_00 := # examples misclassified by both
%  e_11 := # examples correctly classified by both
%  e_10 := # misclassified by 2 but not by 1
%  e_01 := # misclassified by 1 but not by 2
%
%  Under H0, one expects:  e_01 = e_10 = (e_01 + e_10)/2
%  (assuming indepdendent trials)
%
%  The sum of iid random normal variables follows chi-squared
%  distribution:
%     Q = \sum_{i=1}^k  (z_i)^2
%  is distributed according to chi-square w/ k d.o.f.
%  So in the case of k=1, this is just the distribution of
%  the square of a normal random variable.
%
%  References:
%    [1] E Alpayd, Lecture notes for: Introduction to Machine
%    Learning, The MIT Press (V1.1), 2004.
%    [2] https://en.wikipedia.org/wiki/McNemar%27s_test


is_correct_1 = (y_hat_1(:) == y(:));
is_correct_2 = (y_hat_2(:) == y(:));

e_10 = sum(~is_correct_2 & is_correct_1);
e_01 = sum(~is_correct_1 & is_correct_2);
n = e_10 + e_01;


% This code implements an exact p-value for binomial test
% and then corrects for mid-p McNemar test.

b = min(e_10, e_01);
    
if 0
    % This is the brute-force calculation.
    % Computationally it is not a good idea.
    p = 0;
    for ii = 0:b
        p = p + nchoosek(n, ii) * (.5)^ii * (.5)^(n-ii);
    end
    p = 2 * p;
    % correction for mid-p McNemar test
    p = p - nchoosek(n,b) * (.5)^b * (.5)^(n-b);
else
    p = 2*binocdf(b, n, .5) - binopdf(b, n, .5);
end
