function [conjoDx,conjoDy,conjotfH,Nomin1,Denom1,Denom2] = getC(Bn,H)
% [conjoDx,conjoDy,conjotfH,Nomin1,Denom1,Denom2] = getC(Bn,H)
%
% Inputs: 
%        Bn -- blurry and noisy observation f
%        H  -- convolution kernel  
% 
% Outputs:
%      conjoDx -- conjugate of eigenvalues of D^1
%      conjoDy -- conjugate of eigenvalues of D^2
%     conjotfH -- Fourier transform of K'
%       Nomin1 -- Fourier transform of K'*f
%       Denom1 -- |F(D^1)|.^2 + |F(D^2)|.^2
%       Denom2 -- |F(K)|.^2
%

% J. Yang, Jan. 1, 2008

sizeB = size(Bn);

conjoDx = conj(psf2otf([1,-1],sizeB));
conjoDy = conj(psf2otf([1;-1],sizeB));
conjotfH = conj(psf2otf(H,sizeB));
Nomin1 = conjotfH.*fft2(Bn);
Denom1 = abs(conjoDx).^2 + abs(conjoDy ).^2;
Denom2 = abs(conjotfH).^2; 