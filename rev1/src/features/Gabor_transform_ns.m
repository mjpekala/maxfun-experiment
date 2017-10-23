function [coeff,meta] = Gabor_transform_ns(f,A,B)
% GABOR_TRANSFORM computes the semi-discrete Gabor frame coefficients of f 
% using the FFT
%
% INPUT:
% f         (MxN real matrix) function values of f, defined in the spatial domain
% A         (pos int) number of coefficients in the horizontal dimension
% B         (pos int) number of coefficients in the vertical dimesion
%
% OUTPUT
% G         (MxNx(AB) tensor) the Gabor coefficients of in the spatial domain 
%           with respect to a semi-discrete Parseval Gabor frame, where the
%           third index corresponds to the frame index
% meta      ((AB)x2 matrix) information about the Fourier support of each 
%           frame coefficient

% initialization
[N,M] = size(f); % M is num of horizontal points, N is num vertical points
coeff = zeros(N,M,A*B);
meta = zeros(A*B,2);
W = zeros(N,M,A*B);

% Fourier transform of f
F = fftshift(fft2(f));

% lattice step sizes 
s1 = M/(A-1); % horizontal step
s2 = N/(B-1); % vertical step

% create sampling lattice in the Fourier domain
count = 1;
for j = 1:A
    for k = 1:B
        
        % coordinate of the lattice point
        c1 = -M/2+s1*(j-1);
        c2 = -N/2+s2*(k-1);
        
        % update meta
        meta(count,1) = c1;
        meta(count,2) = c2;
        
        % create window centered at (c1,c2) with width (s1,s2)        
        window = eval_window(M,N,c1,c2,s1,s2);
        W(:,:,count) = window;
        
        % compute the frame coefficient via the FFT
        coeff(:,:,count) = ifft2(ifftshift(F.*window));
        
        % update index
        count = count+1;
    end
end

%% test: output should be close to zero since the frame is Parseval
%error = norm(sum(coeff,3)-f)/norm(f)

function window = eval_window(M,N,c1,c2,s1,s2)
% creates a M x N matrix of the function values of g
% g is a trigonometric window centered at c, with widths s1 and s2

% create coordinate mesh in the Fourier domain
[X,Y] = meshgrid(-M/2:M/2-1,-N/2:N/2-1);

% cutoff
cutoff = (abs(X-c1)<s1).*(abs(Y-c2)<s2);

% evaluate the trigonometric function dilated and centered
window = (cos(pi/(2*s1)*(X-c1)) .* cos(pi/(2*s2)*(Y-c2)) .* cutoff).^2;

    