function [Psi, desc] = make_plane_wave(kx, ky, n)

if nargin < 3, n = 128; end


a = inv(2*pi/n);  % make spatial domain span [-pi, pi].
                  % this makes wave numbers more interpretable

[X,Y] = make_image_domain([n,n], a, 'time');

k0 = [kx ky];
ks = sqrt(sum(k0.^2));
theta = atan2d(ky, kx);
Psi = exp(1i*(kx*X + ky*Y));  % exp(i <k0,[x y]>) 
    
desc = sprintf('k0=(%0.1f,%0.1f), |k0|=%0.1f, theta=%0.1f', k0(1), k0(2), ks, theta);
