function fcoeff = Gabor_transform_ns(f,A,B)

% f         (M xN real matrix) a possibly rectangular matrix of function values
%           f is defined the spatial domain
%           M and N are assumed to be even
% A         (pos int) number of Gabor coefficients in the first dimension
% B         (pos int) number of Gabor coefficients in the second dimesion
%           assumed that A and B are both odd or both even
% G         (tensor) the Gabor coefficients of in the spatial domain 
%           with respect to a semi-discrete Parseval Gabor frame
%           tensor is of size M x N x (AB), where the third index 
%           corresponds to the frame index
%
% The third index of G is organized according to the Fourier support of the
% corresponding frame coefficient; namely, they are ordered in a radially 
% increasing and counterclockwise fashion

%% initialization

[M,N] = size(f);
fcoeff = zeros(M,N,A*B);

% Fourier transform of f
F = fftshift(fft2(f));

%% create the tiling coordinates for the frame

% step sizes for frames, which will result in A*B number of frames
s1 = M/(A-1);
s2 = N/(B-1);

if mod(A,2) % A and B are both odd
    
    % r = 0 and r = 1 coefficients
    index = [0, 0; s1, -s2; s1, 0; s1, s2; 0, s2; -s1, s2; -s1, 0; -s1, -s2; 0, -s2];
    
    % "radius"
    for r = 2:(A-1)/2
        
        % east
        for k = -r:r
            vec = [s1*r, s2*k];
            index = [index; vec];
        end
        
        % north
        for k = -(r-1):(r-1)
            vec = [s1*(-r), s2*r];
            index = [index; vec];
        end
        
        % west
        for k = -r:r
            vec = [-s1*r, -s2*k];
            index = [index;vec];
        end
        
        % south
        for k = -(r-1):(r-1)
            vec = [s1*k, -s2*r];
            index = [index; vec];
        end
        
    end
else % A and B are both even
    
    % r = 1 coefficients
    index = [s1/2, -s2/2; s1/2 s2/2; -s1/2, s2/2; -s1/2, -s2/2 ];
    
    % "radius"
    for r = 2:A/2
        
        % east
        for k = -r:(r-1)
            vec = [s1*(r-1/2), s2*(k+1/2)];
            index = [index; vec];
        end
        
        % north
        for k = -(r-1):(r-2)
            vec = [s1*(-k-1/2), s2*(r-1/2)];
            index = [index; vec];
        end
        
        % west
        for k = -r:(r-1)
            vec = [-s1*(r-1/2), s2*(-k-1/2)];
            index = [index; vec];
        end
        
        % south
        for k = -(r-1):(r-2)
            vec = [-s1*(-k-1/2), -s2*(r-1/2)];
            index = [index; vec];
        end
        
    end
 
end

%% create frame and take the convolution simultaneously
for k = 1:A*B
    window = eval_window(M,N,index(k,:),s1,s2);
    fcoeff(:,:,k) = ifft2(ifftshift(F.*window));
end

% test: output should be close to zero since the frame is Parseval
% error = norm(sum(fcoeff,3)-f)/norm(f)

function window = eval_window(M,N,c,s1,s2)
% creates a M x N matrix of the function values of g
% g is a trigonometric window centered at c, with widths s1 and s2

% create coordinate mesh in the Fourier domain
[X,Y] = meshgrid(-M/2:1:M/2-1,-N/2:1:N/2-1);

% cutoff
cutoff = (abs(X-c(1))<s1).*(abs(Y-c(2))<s2);

% evaluate the trigonometric function dilated and centered
window = (cos(pi/(2*s1)*(X-c(1))) .* cos(pi/(2*s2)*(Y-c(2))) .* cutoff).^2;

    
