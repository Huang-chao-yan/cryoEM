function [PSNR, SSIM, UIQI] = quality(imagery1, imagery2)
%==========================================================================
% Evaluates the quality assessment indices for two tensors.
%
% Syntax:
%   [psnr, ssim, fsim] = quality(imagery1, imagery2)
%
% Input:
%   imagery1 - the reference tensor
%   imagery2 - the target tensor

% NOTE: the tensor is a M*N*K array and DYNAMIC RANGE [0, 255]. 

% Output:
%   psnr - Peak Signal-to-Noise Ratio
%   ssim - Structure SIMilarity
%   fsim - Feature SIMilarity

% See also StructureSIM, FeatureSIM
%
% by Yi Peng
% update by Yu-Bang Zheng 11/19/2018
%==========================================================================
Nway = size(imagery1);
PSNR = zeros(Nway(3),1);
SSIM = PSNR;
UIQI = PSNR;
for i = 1:Nway(3)
    PSNR(i) = psnr(imagery2(:, :, i), imagery1(:, :, i),255);
    SSIM(i) = ssim(imagery2(:, :, i), imagery1(:, :, i));
    UIQI(i) = img_qi(imagery1(:, :, i), imagery2(:, :, i));
end
PSNR = mean(PSNR);
SSIM = mean(SSIM);
UIQI = mean(UIQI);

