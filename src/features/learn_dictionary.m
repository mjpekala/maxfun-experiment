function [D param] = learn_dictionary(X, y, varargin)
% LEARN_DICTIONARY   Use SPAMS toolbox to learn a sparse dictionary.
%
%  Note: it is not entirely clear what precise method the authors of
%  [1,2] used to learn their dictionaries.
%
% REFERENCES:
%  [1] Boureau et al. "A Theoretical Analysis of Feature Pooling in Visual Recognition," 2010.
%  [2] Boureau et al. "Learning Mid-Level Features For Recognition," 2010.
%  [3] SPAMS toolbox (http://spams-devel.gforge.inria.fr/)

% mjp, april 2016

parser = inputParser;
parser.addRequired('X', @(X) ndims(X) == 2);
parser.addParameter('nAtoms', 128, @(x) x > 0);
parser.addParameter('verbose', true);
parser.addParameter('k', 5, @(k) k > 1);           % # of folds
parser.addParameter('lambdas', [1e-2 1e-2 1e-1 1 10], @(x) all(x > 0));
parser.addParameter('nonneg', true);
parser.parse(X, varargin{:});

nFolds = parser.Results.k;
nAtoms = parser.Results.nAtoms;
lambdas = parser.Results.lambdas;
useNN = parser.Results.nonneg;


% some additional checks and helper functions
if ~ismember(nAtoms, [128 256 512 1024])
    warning('unexpected # of dictionary atoms - are you sure?');
end

if parser.Results.verbose
    vprintf = @fprintf;
else
    vprintf = @(varargin) 0;
end

shuffle = @(x) x(randperm(length(x)));


% Assign each example to one of nFolds folds.
% (setup for k-fold cross validation)
foldId = assign_folds(y, nFolds);

% sparse coding parameters
% note that we'll update lambda later below
param.K=nAtoms;
param.lambda=NaN;      % for mode 2, the ell_1 coefficient;  will be updated below
param.lambda2 = 0;     % ridge penalty coefficient
param.numThreads=-1; 
param.batchsize=400;
param.verbose=false;
param.iter=1000;       % impacts runtime! XXX: pick this intelligently...
param.mode=2;          % 2 := elastic net.  see documentation
param.warm_restart = true;  % XXX: is this a good idea??
param.pos = useNN;

%-------------------------------------------------------------------------------
% search for best lambda
%-------------------------------------------------------------------------------

if length(lambdas) > 1
    vprintf('[%s]: Searching over %d lambda values and %d folds\n', mfilename, length(lambdas), nFolds);
    perf = zeros(size(lambdas));
    
    for ii = 1:length(lambdas)
        param.lambda = lambdas(ii);
    
        for jj = 1:nFolds
            vprintf('[%s]: trying lambda=%0.2g, fold %d\n', mfilename, param.lambda, jj);
        
            Xtrain = X(:, foldId ~= jj);
            Xvalid = X(:, foldId == jj);
       
            % The SPAMS documentation suggests the _Memory variant may be more
            % efficient for small problems (where memory usage is less of a concern).
            % However, in the one or two examples I looked at, it seems to run 
            % more slowly (and memory does not appear to be an issue).
            %
            % Also, the _Memory variant does not support warm starts,
            % which in theory could be quite useful here (although I have
            % not observed a speedup yet).
            %
            D = mexTrainDL(Xtrain, param);
            %D = mexTrainDL_Memory(Xtrain, param);
 
            % evaluate performance on held-out data
            alpha = mexLasso(Xvalid, D, param);
            f = mean(0.5*sum((Xvalid-D*alpha).^2) + param.lambda*sum(abs(alpha)));
            perf(ii) = perf(ii) + f;
        end
        vprintf('\n');
    end
    
    vprintf('[%s]: obj. func. values: %s\n\n', mfilename, num2str(perf));
    
    % train the actual dictionary using the lambda value with the overall
    % minimum objective function value (averaged over all folds).
    [~,argmin] = min(perf);
    param.lambda = lambdas(argmin);
else
    % No search was requested
    param.lambda = lambdas(1);
end


vprintf('[%s]: learning dictionary using lambda=%0.2g\n', mfilename, param.lambda);
D = mexTrainDL(X, param);
