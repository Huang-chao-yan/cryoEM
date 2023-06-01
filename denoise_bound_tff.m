function [X_den,iter,fun_all,e,re,P,i]=denoise_bound_tff(A,Xobs,lambda,l,u,P_init,pars)
%This function implements the FISTA method for TV denoising problems. This function is used inside the deblurring
% procedure since it uses a warm-start strategy
%
% INPUT
% Xobs ..............................an observed noisy image.
% lambda ........................ parameter
% pars.................................parameters structure
% pars.MAXITER ..................... maximum number of iterations
%                                                      (Default=100)
% pars.epsilon ..................... tolerance for relative error used in
%                                                       the stopping criteria (Default=1e-4)
% pars.print ..........................  1 if a report on the iterations is
%                                                       given, 0 if the  report is silenced
% pars.tv .................................. type of total variation
%                                                      penatly.  'iso' for isotropic (default)
%                                                      and 'l1' for nonisotropic
%  
% OUTPUT
% X_den ........................... The solution of the problem 
%                                            min{||X-Xobs||^2+2*lambda*TV(X)}
% iter .............................  Number of iterations required to get
%                                            an optimal solution (up to a tolerance)
% fun_all ......................   An array containing all the function
%                                             values obtained during the
%                                             iterations


%Define the Projection onto the box
if((l==-Inf)&(u==Inf))
    project=@(x)x;
elseif (isfinite(l)&(u==Inf))
    project=@(x)(((l<x).*x)+(l*(x<=l)));
elseif (isfinite(u)&(l==-Inf))
     project=@(x)(((x<u).*x)+((x>=u)*u));
elseif ((isfinite(u)&isfinite(l))&(l<u))
    project=@(x)(((l<x)&(x<u)).*x)+((x>=u)*u)+(l*(x<=l));
else
    error('lower and upper bound l,u should satisfy l<u');
end

% Assigning parameres according to pars and/or default values
flag=exist('pars');
if (flag&isfield(pars,'MAXITER'))
    MAXITER=pars.MAXITER;
else
    MAXITER=100;
end
if (flag&isfield(pars,'epsilon'))
    epsilon=pars.epsilon;
else
    epsilon=1e-4;
end
if(flag&isfield(pars,'print'))
    prnt=pars.print;
else
    prnt=1;
end
if(flag&isfield(pars,'tv'))
    tv=pars.tv;
else
    tv='iso';
end

[m,n,s]=size(Xobs);
clear P
clear R
% if(isempty(P_init))
    
%     vxi    = GradVol3D(Xobs);% nabla u
%     P      =  -Divz3D(vxi); % (nabla u)'

%%%%%%%%%%%%%%%%%%%%%%%%%frame%%%%%%%%%%% (HCY 211123)
 frame=1;
 Level=2;
%  wLevel=1/2;
% initialization
% [m,n]=size(f);
[Df,~]=GenerateFrameletFilterL(frame);
nD=length(Df);
%Compute the weighted thresholding parameters.
% muLevel=getwThresh(1,wLevel,Level,Df);%%%%
 if(isempty(P_init))
    P=FraDecMultiLevelL(Xobs,Df,Level);  %Wu
    
    R      =  P;
% [conjoDx,conjoDy,conjotfH,Nomin1,Denom1,Denom2] = getC(Xobs,A);
%Denom2=ATA  but we already have ATA

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     R      =  P;
%     P{1}=zeros(m-1,n,s);
%     P{2}=zeros(m,n-1,s);
%     R{1}=zeros(m-1,n,s);
%     R{2}=zeros(m,n-1,s);
else
     P{1}=P_init{1};
     P{2}=P_init{2};
     R{1}=P_init{1};
     R{2}=P_init{2};
 end
tk=1;
tkp1=1;
count=0;
i=0;

D=Xobs;
fval=inf;
fun_all=[];
while((i<MAXITER)&(count<10))
    fold=fval;
    %%%%%%%%%
    % updating the iteration counter
    i=i+1;
    %%%%%%%%%
    % Storing the old value of the current solution
    Dold=D;  
    %%%%%%%%%%
    %Computing the gradient of the objective function
    Pold=P;
    tk=tkp1;
%     D=project(Xobs-lambda*Lforward(R)); %D = Y - 2lambda (nablaT nabla X)
    D=project(Xobs-lambda*Dold); 
%     figure; isosurface(real(D),max(real(D(:)))/5);
%     Q=Ltrans(D);
%     Q   = -Divz3D(GradVol3D(D));
    Q = FraDecMultiLevelL(D,Df,Level);  %Wu;
    

    %%%%%%%%%%
    % Taking a step towards minus of the gradient
    %      P=R+1/(8*lambda)*Q;  % Cryo EM TV
    
%      P{1}=R{1}+1/(8*lambda)*Q{1};  % original TV

%      P{2}=R{2}+1/(8*lambda)*Q{2};
    
         for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                    P{ki}{ji,jj}=R{ki}{ji,jj}+1/(8*lambda)*Q{ki}{ji,jj};
                end
            end
         end
        
     
     
    %%%%%%%%%%
    % Peforming the projection step
%     switch tv
%         case 'iso'
% %             A=[P{1};zeros(1,n,s)].^2+[P{2},zeros(m,1,s)].^2;
%                 A=norm(P(:,:,1));
%                 for kk=1:s-1
%                     B=norm(P(:,:,kk+1));
%                     A=A+B;
%                 end
%      
% %             A=sqrt(max(A,1));
%             P = P./A;
% %             P{1}=P{1}./A(1:m-1,:,s);
% %             P{2}=P{2}./A(:,1:n-1,s);
%         case 'l1'
%             P{1}=P{1}./(max(abs(P{1}),1));
%             P{2}=P{2}./(max(abs(P{2}),1));
%         otherwise
%             error('unknown type of total variation. should be iso or l1');
%     end

    %%%%%%%%%%
    %Updating R and t    here R is Y in the algorithm (HCY 211117)
    tkp1=(1+sqrt(1+4*tk^2))/2;
    
    
%     R = P+(tk-1)/(tkp1)*(P-Pold);
    
%     R{1}=P{1}+(tk-1)/(tkp1)*(P{1}-Pold{1});
%     R{2}=P{2}+(tk-1)/tkp1*(P{2}-Pold{2});



    
         for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                    R{ki}{ji,jj}=P{ki}{ji,jj}+(tk-1)/(tkp1)*(P{ki}{ji,jj}-Pold{ki}{ji,jj});
                end
            end
         end
        

    
    re(i)=norm(D(:)-Dold(:),'fro')/norm(D(:),'fro');
    
    if (re(i)<epsilon)
        count=count+1;
    else
        count=0;
    end
%     C=Xobs-lambda*Lforward(P);
%     PQ   = -Divz3D(GradVol3D(P));
%     C=Xobs-lambda*(PQ);



%     C = Xobs-lambda*(P);  
         for ki=1:Level
            for ji=1:nD-1
                for jj=1:nD-1
                   C{ki}{ji,jj}=Xobs-2*lambda*P{ki}{ji,jj};
                end
            end
         end
         
    
    
     PC=project(C);
    fval=-norm(C{1}{1}(:,:,35)-PC{1}{1}(:,:,35),'fro')^2+norm(C{1}{1}(:,:,35),'fro')^2;
    fun_all=[fun_all;fval];
    e(i)=norm(D(:,:,35)-Dold(:,:,35),'fro')/norm(Dold(:,:,35),'fro');
    paner   = norm(D(:)-Dold(:))/norm(Dold(:));

    if(prnt)
        fprintf('iter= %5d value = %10.10f %10.10f  %15.7f ',i,fval,e(i),paner);
        if (fval>fold)
            fprintf('  *\n');
        else
            fprintf('   \n');
        end
    end
end
X_den=D;
iter=i;

