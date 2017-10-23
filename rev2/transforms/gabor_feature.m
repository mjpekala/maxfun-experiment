function G = gabor_feature(f,A,B,R)
% GABOR_TRANSFORM computes the semi-discrete Gabor frame coefficients of an
% image using the Fourier transform
%
% INPUT:
% f         (MxN real array) samples a function
% A         (pos int) number of coefficients in the horizontal dimension
% B         (pos int) number of coefficients in the vertical dimesion
% R         (pos int) optional parameter
%
% OUTPUT
% G         (MxNx(AB) tensor) the Gabor coefficients of in the spatial domain 
%           with respect to a semi-discrete Parseval Gabor frame, where the
%           third index corresponds to the frame index
% meta      ((AB)x2 matrix) information about the ordering of the
%           coefficients

% initialization
if nargin <= 3
    R = 1; % no apriori frequency concentration
end

% initialize
[N,M] = size(f); % M is num of horizontal points, N is num vertical points

% unitary discrete Fourier transform of f
F = fftshift(fft2(f))/sqrt(M*N);

% lattice step sizes 
u = M/(A+1)/R; % horizontal step size
v = N/(B+1)/R; % vertical step size

% size of coefficients
U = 2*ceil(u)-1;
V = 2*ceil(v)-1;
G = zeros(N,M,A*B);

% create sampling lattice in the Fourier domain
count = 1;
for j = 1:A
    for k = 1:B
        
        % coordinate of the lattice point
        m = -(M-1)/2/R + u*j;
        n = -(N-1)/2/R + v*k;
        
        % create window centered at (m,n) with side lengths (u,v)        
        [win, supp] = eval_cutoff(M,N,m,n,u,v);
        
        % compute the gabor coefficient via the Fourier transform
        F_loc = F.*win;
        %F_loc = F_loc(supp(1):supp(2),supp(3):supp(4));
        G(:,:,count) = ifft2(ifftshift(F_loc))*sqrt(M*N);
        
        % update index
        count = count+1;
    end
end

%% test: output should be close to zero since the frame is Parseval
%error = norm(sum(coeff,3)-f)/norm(f)

function [window, support] = eval_cutoff(M,N,m,n,u,v)
% creates a M x N matrix of the function values of g
% g is a trigonometric window centered at (m,n), with side lengths (u,v)

% create coordinate mesh in the Fourier domain
[X,Y] = meshgrid(-(M-1)/2:(M-1)/2,-(N-1)/2:(N-1)/2);

% cutoff
cutoff = (abs(X-m)<u).*(abs(Y-n)<v);

% evaluate the dilated and centered trigonometric function 
window = (cos(pi/(2*u)*(X-m)) .* cos(pi/(2*v)*(Y-n)) .* cutoff);

% find its support
[row,col] = find(window>0);
support = [min(row), max(row),min(col), max(col)];

    