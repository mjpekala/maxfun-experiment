function X = gabor_edge_feature(I,S,R,isreal)
% GABOR_EDGE_FEATURE extracts edges of an image at various angles
% S is number of Fourier scales
% R is number of rotations
% S and R are assumed to be an integer

% default is real images
if nargin < 4
    isreal = 'real';
end

% coordinates
[N,M] = size(I);
[x,y] = meshgrid(-(M-1)/2:(M-1)/2,-(N-1)/2:(N-1)/2);
[theta,rho] = cart2pol(x,y);
theta_new = theta + 2*pi*(theta<0);

% parameters
A = floor(min((N-1)/2,(M-1)/2)/(S+1));
if strcmp('real',isreal)
    B = pi/R;
else
    B = 2*pi/R;
end

% spectral data
I = double(I);
F = fftshift(fft2(I));

X = zeros(N,M,S*R);
count = 1;

for s = 2:S+1
    % radial cutoff
    if s == 0
        rho_cutoff = (rho <= A/3) + eval_sigmoid(rho,A/3,2*A/3,'cos');
    else
        left = eval_sigmoid(rho,s*A-2*A/3,s*A-A/3,'sin');
        right = eval_sigmoid(rho,s*A+A/3,s*A+2*A/3,'cos');
        rho_cutoff = (abs(rho-s*A)<=A/3) + left + right;
    end
    
    % angular cutoff
    for r = 0:R-1
        if r == 0
            right = eval_sigmoid(theta,B/3,2*B/3,'cos');
            left =  eval_sigmoid(theta,-2*B/3,-B/3,'sin');
            theta_cutoff = (abs(theta) <= B/3) + right + left;
        else
            % rotate coordinates
            right = eval_sigmoid(theta_new,r*B+B/3,r*B+2*B/3,'cos');
            left = eval_sigmoid(theta_new,r*B-2*B/3,r*B-B/3,'sin');
            theta_cutoff = (abs(theta_new-r*B) <= B/3) + right + left;
        end
        
        % compute the coefficient
        X(:,:,count) = real(ifft2(ifftshift(F.*rho_cutoff.*theta_cutoff)));
        count = count+1;
        %figure; imagesc(rho_cutoff.*theta_cutoff);
    end
end

function sigmoid = eval_sigmoid(X,a,b,str)
% X vector or array of coordinates
% creates a sigmoid function S evaluated on X such that S(a) = 0 and S(b) = pi/2

X = (X-a)/(b-a);            % shift and normalize
X_supp = (X>0).*(X<1);      % support of X

% set values off of S to 5, speeds up calculation
Y = 5*(X<=0) + 5*(X>=1) + X_supp.*X;

F = (pi/2)./(1+exp((2*Y-1)./((Y-1).*Y)));

if strcmp(str,'cos')
    sigmoid = X_supp.*cos(F);
elseif strcmp(str,'sin')
    sigmoid = X_supp.*sin(F);
end

