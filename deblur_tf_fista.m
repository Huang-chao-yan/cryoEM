function [X_out,fun_all]=deblur_tf_fista(A,xBobs,ATA,ATg,center,lambda,l,u,ph)
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
    MAXITER=100;
end
if(flag&isfield(pars,'fig'))
    fig=pars.fig;
else
    fig=0;
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
% if (nargout==1)
    fun_all=1e-5;
% end

[m,n,s]=size(xBobs);
n       = size(xBobs,1); 
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
%          trans=@(X) 1/sqrt(m*n)*fft2(X);
%          itrans=@(X) sqrt(m*n)*ifft2(X);
%         % computng the eigenvalues of the blurring matrix         
%         Sbig=fft2(circshift(Pbig,1-center));
%     otherwise
%         error('Invalid boundary conditions should be reflexive or periodic');
% end

Sbig=ATA;

% computing the two dimensional transform of Bobs
% Btrans=trans(Bobs); %FFT (HCY211117)

%The Lipschitz constant of the gradient of ||A(X)-Bobs||^2
aa = zeros(size(Bobs));
for ii=1:n
     aa(:,:,ii) = eye(n);
end
La = ATA(aa);
lip = zeros(1,n);
for kk=1:n
    [uu,dd,vv] = svd(La(:,:,kk));
    lip(kk) = max(max(dd));
end
Lip = max(lip);
L = 5*Lip;
% L=16*lambda;%2lambda_max(ATA)
% L=3*L;

% fixing parameters for the denoising procedure 
clear parsin
%parsin.MAXITER=denoiseiter;
parsin.epsilon=1e-5;
parsin.print=0;
parsin.tv=tv;


count_inner_error=0;


% initialization
% X_iter=Bobs;
load('panresult.mat')
X_iter = OutPuto;
Y=X_iter;
t_new=1;
errr=zeros(1,MAXITER);
SolRE  = 1e-6; 
re = 1; 
reph=1;

fprintf('***********************************\n');
fprintf('*   Solving with FISTA      **\n');
fprintf('***********************************\n');
fprintf('#iter           fun-val             tv          denoise-iter           relative-dif        ER          PSNR\n===============================================\n');
% for i=1:MAXITER
i=2;
[MPSNRALL, SSIMALL, FSIMALL] = quality(real(X_iter)*255, double(ph)*255);
PSNR(2) = MPSNRALL;
 lambda=lambda/200;
while (i<MAXITER)&&(reph>SolRE)
%     k = k+1;  
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
    
     if PSNR(i-1)>=PSNR(i)
        [Z_iter,iter,fun_denoise,e,re,P,ik]=denoise_bound_tf(A,Y,2*lambda/L,l,u,[],parsin);
    else
        [Z_iter,iter,fun_denoise,e,re,P,ik]=denoise_bound_tff(A,Y,2*lambda/L,l,u,[],parsin); %% the core function  (HCY 211117)
    end
    % in the initial, first generate the P, i.e., the [Dx, Dy]
    % then, we have the gradient operator (HCY 211117)
    % so I need to write a denoise code, then to generate a deblur code
    if count_inner_error
        ee = zeros(ik);
        ere = zeros(ik);
        ee = e;
        ere = re;
        ero='./e';
        save([ero/'e_',num2str(i),'.mat'],'ee')
        save([ero/'er_',num2str(i),'.mat'],'ere')
    end
    
    % Compute the total variation and the function value and store it in
    % the function values vector fun_all if exists.
%     t=tlv(Z_iter,tv); % we need to change the corresponding regularzer here (HCY 211117)
    
    t(i)=tff(Z_iter);
%     Ax=A*(Z_iter);
%     alpha = 1e-5;
%     Treg  = @(x)(ATA(x) + x/alpha);  %A
%     Ax = Treg(Z_iter);
    Ax = A*Z_iter;
    fun_val=norm(Ax(:)-xBobs(:),'fro')^2+lambda*t(i); % || A x-B||^2+2lambda TV, the energy  (HCY 211117)
    
    F(i)=fun_val;
    
    if (mon==0)
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
    
    errr(i)=norm(X_iter(:,:,35)-X_old(:,:,35),'fro')/norm(X_old(:,:,35),'fro');
    re   = norm(X_iter(:)-X_old(:))/norm(X_old(:));
    reph   = norm(X_iter(:)-ph(:))/norm(ph(:));
    
    ER_pan(i) = re;
     ER_ph(i) = reph;
     [MPSNRALL, SSIMALL, FSIMALL] = quality(real(X_iter)*255, double(ph)*255);

     PSNR(i+1) = MPSNRALL;
    % printing the information of the current iteration
    fprintf('%3d    %15.7f %15.7f           %3d       %15.7f      %15.7f         %15.7f \n',i,fun_val,t(i),iter,errr(i),reph,MPSNRALL);

    if (fig)
        figure; isosurface(real(X_iter),max(real(X_iter(:)))/5);
    end
    i = i+1;
    
end

X_out.u=X_iter;
X_out.er=errr;
plot(PSNR)
end


function t= tf(u)
    
 frame=1;
 Level=2;
 wLevel=1/2;
[Df,R]=GenerateFrameletFilter(frame);
nD=length(Df);
%Compute the weighted thresholding parameters.
% muLevel=getwThresh(1,wLevel,Level,Df);%%%%
C=FraDecMultiLevel(u,Df,Level); 
 J2=zeros(size(u));   
    
 for k=1:Level
    for ii=1:nD-1
        for j=1:nD-1
              J2=J2+((C{k}{ii,j}));
        end
    end
 end
    t=norm(J2(:,:,1));
end

function t= tff(u)
    dim = size(u);
 frame=1;
 Level=2;
 wLevel=1/2;
[Df,R]=GenerateFrameletFilter(frame);
nD=length(Df);
%Compute the weighted thresholding parameters.
% muLevel=getwThresh(1,wLevel,Level,Df);%%%%
 C         = Fold( FraDecMultiLevel(Unfold(u,size(u),3),Df,Level) ,  [dim(1:2),dim(3)*size(Df,1)*Level],3);

 J2=zeros(size(u));   
    
 len =1;
 
%  for k=1:Level
%     for ii=1:nD-1
%         for j=1:nD-1
%               J2=J2+((C{k}{ii,j}));
%         end
%     end
%  end
   
    nfilter =1;

    for ki=1:Level
        for ii=1:nD
            J2(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)= J2(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len)+C(:,:,len*(ki-1)*nD+(ii-1)*len+1: len*(ki-1)*nD+ii*len);
        end
        
    end
     t=norm(J2(:,:,1));
    
end
