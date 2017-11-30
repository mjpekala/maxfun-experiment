function mcnemar_multiclass(y_hat_1, y_hat_2, y_true, ...
                            algo_1_name, algo_2_name)
% MCNEMAR_MULTICLASS  A wrapper around mcnemar() that reports 
%                     per-class results.
%
% Note: you may want to perform some kind of correction for the
%       multiplicity effect (e.g. see [1,2]) when analyzing these
%       results.
%
%  REFERENCES
%    [1] Salzberg "On comparing classifiers: Pitfalls to avoid and
%                  a recommended approach." 1997.
%    [2] https://en.wikipedia.org/wiki/False_discovery_rate
%

% mjp, october 2016

if nargin < 5, algo_2_name = 'algorithm 2'; end
if nargin < 4, algo_1_name = 'algorithm 1'; end

yAll = sort(unique(y_true(:)));

fprintf('-------------------------------------------------\n');
fprintf('[%s]: comparing %s  with %s using McNemars test\n', ...
        mfilename, algo_1_name, algo_2_name);

fprintf('  e_10 := # %s correct and %s incorrect\n', ...
        algo_1_name, algo_2_name)
fprintf('  e_01 := # %s incorrect and %s correct\n', ...
        algo_1_name, algo_2_name)
fprintf('  p    := p-value from McNemars test\n');
fprintf('  n    := # of test instances with true label yi\n');
fprintf('-------------------------------------------------\n');

for ii = 1:length(yAll), yi = yAll(ii);
    idx = (y_true(:) == yi);
    [p, e_10, e_01] = mcnemar(y_hat_1(idx), ...
                              y_hat_2(idx), ...
                              y_true(idx));
    fprintf('  y=%3d, n=%4d, e_10=%3d, e_01=%3d, p=%0.4f\n', ...
            yi, sum(idx), e_10, e_01, p);
end
