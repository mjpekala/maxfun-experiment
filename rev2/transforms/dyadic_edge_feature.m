function X = dyadic_edge_feature(I,S,R,isreal)
% GABOR_EDGE_FEATURE extracts edges of an image at various angles
% S is number of dyadic scales
% R is number of rotations
% S and R are assumed to be an integer

% default is real images
if nargin < 4
    isreal = 'real';
end

% coordinates
[N,M] = size(I);
if min((M-1)/2,(N-1)/2)<4
    error('image is probably too small for this transform to be good')
end
[x,y] = meshgrid(-(M-1)/2:(M-1)/2,-(N-1)/2:(N-1)/2);
[theta,rho] = cart2pol(x,y);
theta_new = theta + 2*pi*(theta<0);

% parameters
scales = ceil(log2(min((N-1)/2,(M-1)/2)));   % highest scale
if scales-S<=0
    error('number of scales is too large')
end

if strcmp('real',isreal)
    B = pi/R;
else
    B = 2*pi/R;
end
X = zeros(N,M,R*S);

% spectral data
I = double(I);
F = fftshift(fft2(I));

% lowest scale
% rho_cutoff = (rho <= 2^A) + eval_sigmoid(rho,2^A,2^A+2^A/3,'cos');
% X(:,:,1) = real(ifft2(ifftshift(F.*rho_cutoff)));

% higher scales
count = 1;
for s = scales-S+1:scales
    % radial cutoff
    left = eval_sigmoid(rho,2^(s-1),2^(s-1)+2^(s-1)/3,'sin');
    right = eval_sigmoid(rho,2^s,2^s+2^s/3,'cos');
    center = (rho >= 2^(s-1)+2^(s-1)/3) .* (rho <= 2^s);
    rho_cutoff = left + center + right;
    
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