function [X_out,fun_all]=deblur_tv_fista(A,Bobs,ATA,ATg,center,lambda,l,u)
%ctf_obsimg,T,back_projs,[],eta0,-Inf,Inf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function implements FISTA for solving the linear inverse problem with 
% the total variation regularizer and either reflexive or periodic boundary
% conditions
%
% Based on the paper
% Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained
% Total Variation Image Denoising and Deblurring Problems"
% -----------------------------------------------------------------------
% Copyright (2008): Amir Beck and Marc Teboulle
% 
% FISTA is distributed under the terms of 
% the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% INPUT
%
% Bobs............................. The observed image which is blurred and noisy
% P .................................... PSF of the blurring operator
% center ......................  A vector of length 2 containing the center
%                                           of the PSF
% lambda ...................... Regularization parameter
% l ................................... Lower bound on each of the components
%                                         if no lower bound exists then l=-Inf
% u..................................... Upper bound on each of the components
%                                          if no upper bound exists then u=Inf
% pars.................................Parameters structure
% pars.MAXITER ..................... maximum number of iterations
%                                                      (Default=100)
% pars.fig ............................... 1 if the image is shown at each
%                                                      iteration, 0 otherwise (Default=1)
% pars.BC .................................. boundary conditions.
%                                                      'reflexive' (default)  or 'periodic
% pars.tv .................................. type of total variation
%                                                      penatly.  'iso' for isotropic (default)
%                                                      and 'l1' for nonisotropic
% pars.mon ............................... 1 if a monotone version of the
%                                                      algorithm is used an 0 otherwise (default)
% pars.denoiseiter .......... number of iterations of the denoising inner
%                                                      problem (default=10)
% OUTPUT
% 
% X_out ......................... Solution of the problem
%                                          min{||A(X)-Bobs||^2+2*lambda*TV(
%                                          X): l <=X_{ij}<=u}
% fun_all .................... Array containing all function values
%                                          obtained in the FISTA method


% Assigning parameres according to pars and/or default values
flag=exist('pars');
if (flag&isfield(pars,'MAXITER'))
    MAXITER=pars.MAXITER;
else
    MAXITER=10;
end
if(flag&isfield(pars,'fig'))
    fig=pars.fig;
else
    fig=1;
end
     
    
if (flag&isfield(pars,'BC'))
    BC=pars.BC;
else
    BC='reflexive';
end
if (flag&isfield(pars,'tv'))
    tv=pars.tv;
else
    tv='iso';
end
if (flag&isfield(pars,'mon'))
    mon=pars.mon;
else
    mon=0;
end
if (flag&isfield(pars,'denoiseiter'))
    denoiseiter=pars.denoiseiter;
else
    denoiseiter=10;
end


% If there are two output arguments, initalize the function values vector.
if (nargout==1)
    fun_all=[];
end

[m,n,s]=size(Bobs);
n       = size(Bobs,1); 
Bobs=zeros(n,n,n);
% Pbig=padPSF(P,[m,n,s]);

% switch BC
%     case 'reflexive'
%         trans=@(X)dct2(X);
%         itrans=@(X)idct2(X);
%         % computng the eigenvalues of the blurring matrix         
%         e1=zeros(m,n);
%         e1(1,1)=1;
%         Sbig=dct2(dctshift(Pbig,center))./dct2(e1);
%     case 'periodic'
         trans=@(X) 1/sqrt(m*n)*fft2(X);
         itrans=@(X) sqrt(m*n)*ifft2(X);
%         % computng the eigenvalues of the blurring matrix         
%         Sbig=fft2(circshift(Pbig,1-center));
%     otherwise
%         error('Invalid boundary conditions should be reflexive or periodic');
% end

Sbig=ATA;

% computing the two dimensional transform of Bobs
Btrans=trans(Bobs); %FFT (HCY211117)

%The Lipschitz constant of the gradient of ||A(X)-Bobs||^2
L=16*lambda;%*max(max(max(abs(Sbig(Bobs)).^2)));


% fixing parameters for the denoising procedure 
clear parsin
parsin.MAXITER=denoiseiter;
parsin.epsilon=1e-5;
parsin.print=0;
parsin.tv=tv;

% initialization
X_iter=Bobs;
Y=X_iter;
t_new=1;

fprintf('***********************************\n');
fprintf('*   Solving with FISTA      **\n');
fprintf('***********************************\n');
fprintf('#iter  fun-val            tv             denoise-iter         relative-dif\n===============================================\n');
for i=1:MAXITER
    % store the old value of the iterate and the t-constant
    X_old=X_iter;
    t_old=t_new;
    % gradient step
%     D=Sbig(trans(Y))-Btrans; %D=A.* Y - B  (HCY 211117)
%     Y=Y-2/L*itrans(conj(Sbig).*D); % Y=Y-2/L (AT).*D  (HCY 211117)
%     Y=real(Y);       
    Y = Y-2/L.*(ATA(Y)-ATg);
    
    % the three lines means Y=Y-2/L \nabla f(x)     (HCY 211117)
    % here Y is the x_0, so they are like    
    %x_k = x_{k-1} - t_k \nabla f(x_{k-1})   (HCY 211117)
     
    %invoking the denoising procedure 
    if (i==1)
        [Z_iter,iter,fun_denoise,P]=denoise_bound_TV(Y,2*lambda/L,l,u,[],parsin);
    else
        [Z_iter,iter,fun_denoise,P]=denoise_bound_TV(Y,2*lambda/L,l,u,P,parsin); %% the core function  (HCY 211117)
    end
    % in the initial, first generate the P, i.e., the [Dx, Dy]
    % then, we have the gradient operator (HCY 211117)
    % so I need to write a denoise code, then to generate a deblur code
    
    % Compute the total variation and the function value and store it in
    % the function values vector fun_all if exists.
    t=tlv(Z_iter,tv); % we need to change the corresponding regularzer here (HCY 211117)
    Ax=A*(Z_iter);
    fun_val=norm(Ax(:,:,1)-Bobs(:,:,1),'fro')^2+2*lambda*t; % || A x-B||^2+2lambda TV, the energy  (HCY 211117)
    if(mon==0)
        X_iter=Z_iter;
    else
        if(i>1)
            fun_val_old=fun_all(end);
            if(fun_val>fun_val_old)
                X_iter=X_old;
                fun_val=fun_val_old;
            else
                X_iter=Z_iter;
            end
        end
    end
    if (nargout==1)
        fun_all=[fun_all;fun_val];
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%HERE%%%%%%%%%%%%%%%%%%%%%%% (HCY 211117)
    %updating t and Y
    t_new=(1+sqrt(1+4*t_old^2))/2;
    Y=X_iter+t_old/t_new*(Z_iter-X_iter)+(t_old-1)/t_new*(X_iter-X_old);
    %  y_{k+1} = x_k + (t_k-1)/t_{k+1} *(x_k -x_{k-1}) in the paper 
    %  y_{k+1} = x_k +(t_k)/t_{k+1}*(z_k -x_{k-1}) + (t_k-1)/t_{k+1}*(x_k -x_{k-1}) in the code
    %  because there are 2 cases, Z_iter=X_iter or X_iter=X_old 
    %  in all, Z_iter=X_iter, the core function is in line 159 denoise_bound_init.m
    %  so in general, it is the same as the paper   (HCY 211117)
    
    % printing the information of the current iteration
    fprintf('%3d    %15.5f %15.5f           %3d                  %15.5f\n',i,fun_val,t,iter,norm(X_iter(:,:,1)-X_old(:,:,1),'fro')/norm(X_old(:,:,1),'fro'));
    
    if (fig)
        figure(314); isosurface(real(X_iter),max(real(X_iter(:)))/5);
    end
end

X_out=X_iter;
end

function out=tlv(X,type)
%This function computes the total variation of an input image X
%
% INPUT
%
% X............................. An image
% type .................... Type of total variation function. Either 'iso'
%                                  (isotropic) or 'l1' (nonisotropic)
% 
% OUTPUT
% out ....................... The total variation of X.
[m,n,s]=size(X);
P=Ltrans(X);

switch type
    case 'iso'
        D=zeros(m,n,s);
        D(1:m-1,:,:)=P{1}.^2;
        D(:,1:n-1,:)=D(:,1:n-1)+P{2}.^2;
        out=sum(sum(sum(sqrt(D))));
    case 'l1'
        out=sum(sum(abs(P{1})))+sum(sum(abs(P{2})));
    otherwise
        error('Invalid total variation type. Should be either "iso" or "l1"');
end
end

