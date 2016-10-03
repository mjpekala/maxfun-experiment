%
% An attempt to reproduce (and possibly extend) the result shown in
% figure 1 of [1].  Instead of using theoretical values, we are
% using monte carlo approximations.  Once we obtain a nicer closed
% form representation for the SOS pooling this script can be
% replaced.
%
% The SOS pooling was orginally conceived as a way to bound
% the performance of MAXFUN pooling (which is what we ultimately
% care about).  It could, of course, also be considered as
% an algorithm in its own right.
%
% REFERENCES
%  [1] Boureau et al. "A Theoretical Analysis of Feature Pooling in
%      Visual Recognition," 2010.

% mjp, april 2016


%% Experiment Parameters
nTrials = 500;

m_.nMax = 3000;
m_.mfFloor = [1 5 10 30];  % 'c' 10, 'b', 30
m_.subfig = 'c';

switch(m_.subfig)
  case {'a'}
    m_.alpha1 = 0.4;
    m_.alpha2 = 0.2;
    m_.xlim = [0 26];
    
  case {'b'}
    m_.alpha1 = 1.1e-2;
    m_.alpha2 = 5.1e-3;
    m_.xlim = [0 1000];
   
  case {'c'}
    m_.alpha1 = 1.1e-2;
    m_.alpha2 = 1.1e-4;
    m_.xlim = [0 3000];
    
  otherwise
    error('unknown subfigure id');
end


%% Generate monte-carlo samples

% X_i store monte carlo samples for average pooling
X1 = zeros(m_.nMax, nTrials);
X2 = zeros(size(X1));

% Y_i store monte carlo samples for max pooling
Y1 = zeros(size(X1));
Y2 = zeros(size(Y1));

% Z_i store monte carlo samples for SOS pooling
Z1 = zeros([size(Y1) numel(m_.mfFloor)]);
Z2 = zeros([size(Y1) numel(m_.mfFloor)]);

tic
for ii = 1:nTrials
    % Sample from Bernoulli distribution
    rv = rand(m_.nMax,1);
    coins1 = (rv < m_.alpha1);
    coins2 = (rv < m_.alpha2);
 
    % avg pooling
    X1(:,ii) = cumsum(coins1) ./ (1:length(coins1))';
    X2(:,ii) = cumsum(coins2) ./ (1:length(coins2))';
 
    % max pooling
    Y1(:,ii) = cummax(coins1);
    Y2(:,ii) = cummax(coins2);
 
    % SOS pooling
    %
    for kk = 1:numel(m_.mfFloor)
        mfFloor = m_.mfFloor(kk);
        topk1 = [];
        topk2 = [];
        
        for jj = 1:m_.nMax
            % This might be more elegantly implemented with a selection 
            % algorithm; however, since this is just a back-of-the-envelope
            % calculation, just do what is most expedient.
            
            if jj <= mfFloor
                % just take all the coins
                topk1 = coins1(1:jj);   mf1 = sum(topk1) / mfFloor;
                topk2 = coins2(1:jj);   mf2 = sum(topk2) / mfFloor;
            else
                % update the sets containing the k largest items
                if coins1(jj) > topk1(end)
                    topk1(end) = coins1(jj);
                    topk1 = sort(topk1, 'descend');
                    mf1 = sum(topk1) / mfFloor;
                end

                if coins2(jj) > topk2(end)
                    topk2(end) = coins2(jj);
                    topk2 = sort(topk2, 'descend');
                    mf2 = sum(topk2) / mfFloor;
                end
            end

            Z1(jj,ii,kk) = mf1;
            Z2(jj,ii,kk) = mf2;
        end
    end
end
toc

avgpool = mean_variance_ratio(X1, X2);
maxpool = mean_variance_ratio(Y1, Y2);
mfpool = mean_variance_ratio(Z1, Z2);

xv = 1:m_.nMax;

figure;
plot(xv, maxpool.phi, 'r', ...
     xv, maxpool.sigma1 + maxpool.sigma2, 'b--', ...
     xv, maxpool.psi, 'g:', ...
     xv, avgpool.psi, 'm-.', ...
     'LineWidth', 2);
xlabel('pool cardinality');
legend('\phi_{max}', '\sigma_1 + \sigma_2', '\psi_{max}', '\psi_{avg}', 'Location', 'East');
xlim(m_.xlim);
saveas(gcf, ['Fig1_MC_' m_.subfig '_1.eps'], 'epsc');


figure;
plot(xv, maxpool.psi, 'g:', ...
     xv, avgpool.psi, 'm-.', ...
     xv, mfpool.psi(:,:,2), 'b', ...
     xv, mfpool.psi(:,:,3), 'c', ...
     xv, mfpool.psi(:,:,4), 'r', ...
     'LineWidth', 2);
xlabel('pool cardinality');
legend('\psi_{max}', '\psi_{avg}', ...
       ['\psi_{mf} (' num2str(m_.mfFloor(2)) ')'], ...
       ['\psi_{mf} (' num2str(m_.mfFloor(3)) ')'], ...
       ['\psi_{mf} (' num2str(m_.mfFloor(4)) ')'], ...
       'Location', 'NorthWest');
saveas(gcf, ['Fig1_MC_' m_.subfig '_2.eps'], 'epsc');
