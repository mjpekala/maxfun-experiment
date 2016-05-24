function [wavelett]=wavelet_feature(inputim,J)
%create grid used for computing wavelets. Getting wavelet from intervals
%[-a,a].
%inputim is the input image stack.
%J is the level of desired wavelet transform. J is an integer.
%output wavelett is L1xJxHxW. L1: number of images. J: level of wavelet
%transform. HxW: dimension of the input images.
%One can choose which wavelet function to use. Here I only implemented the
%Morlet wavelet. Improvements can be made to include a various choice of
%wavelets.
%can be modified to add rotations.
%It is uncertain where the original image functions lie in the plane, so I
%chose x and y to be both in [-a,a]. 
%Yiran Li May 23rd, 2016.
[H,W,L1]=size(inputim);
a=10;
base1=linspace(-a,a,H);
base2=linspace(-a,a,W);
Grid=zeros(2,H,W);
Grid(1,:,:)=ones(H,1)*base2;
Grid(2,:,:)=transpose(ones(W,1)*base1);
%NormG=squeeze(sqrt(Grid(1,:,:).^2+Grid(2,:,:).^2));

%create dilated coefficients for wavelets.
% L=8;
% 
% utheta1=zeros(2,L,H,W);
% for l=1:L
%     theta=l*pi/L;
%     Rotation=[cos(theta) -sin(theta);sin(theta) cos(theta)];
%     for i=1:W
%     utheta1(:,l,:,i)=Rotation*squeeze(Grid(:,:,i));
%     end
% 
% end
%%
%u gives the dilated coordinates, 
%J=10;
u=zeros(2,J,H,W);
for i=0:J-1
    u(:,i+1,:,:)=Grid.*2^(-i);
end

%%
%utheta1=[];

%compute the constant C in Morlet wavelet.
%C=sum(sum(exp(-(Grid(1,:,:).^2+Grid(2,:,:).^2/0.5)/2).*exp(i*3*pi/4*Grid(1,:,:))))/sum(sum(exp(i*3*pi/4*Grid(1,:,:))));

%compute wavelets at each dilated and rotated point. Use Morlet wavelet.
%C=0. C needs to be computed so that integral of psi =0.
psi=zeros(size(u(1,:,:,:)));
for j=1:J
psi(:,j,:,:)=2^(-(j-1))*exp(-(u(1,j,:,:).^2+u(2,j,:,:).^2/0.5)/2).*exp(1i*3*pi/4*u(1,j,:,:));
end

psi=squeeze(psi);
%%
%Use FFT to compute convolution of Densities and psi(wavelet).
%compute psi hat and densities hat, the Fourier transform of the two
%functions.
psih=fft2(psi);
densitiesh=fft2(inputim);
%%
%psi=[];
%multipy two functions and use inverse FFT to get convolution of Densities
%and psi.
%%
%L1=92;
%L1=10;
waveletth=zeros(L1,J,H,W);
for j=1:L1
    for i=1:J
 waveletth(j,i,:,:)=squeeze(densitiesh(:,:,j)).*squeeze(psih(i,:,:));
    end
end
%%
%densitiesh=[];
%%

%apply inverse fft to get wavelet transform of rho(densities).
wavelett=ifft2(waveletth);
