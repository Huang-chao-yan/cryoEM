function OutPut = CryoEM_TF_ama(A,g,T,b,eta,eta0,sigma,voltrue)


MaxIter = 100;             
SolRE  = 1e-4; 
tol = 1e-4; 
n       = size(b,1); 
re     = 1; k = 0; 
s      = 0.5; %s = alpha
t      = 8e-2; %1/32/s;  % t = beta


%Heig = CryoEMToeplitz2CirEig(H);
%A       = fftnCryoEMkernel(H); 

start_time = cputime;
vol    = zeros(n,n,n); 
u = vol;
% vxi    = GradVol3D(vol);
%beta   = t * alpha; 
alpha  = 0.05;%1e-1; 

while (k<MaxIter)%&&(re(k)>SolRE)
    k = k+1;  
    
    %% update the primal variable 
%     v      = vol - t * Divz3D(vxi);  %v=vk-t div eta k; section 3.3.1 of the corresponding paper (HCY 211116)

    tmp    = conj(u) .* (T(u) - 2*b);
    Jdata  = real(sum(tmp(:)) + eta0); 
    if Jdata <= eta
        volnew = u; %alpha = 0; 
    else        
        [volnew,alpha] = CryoEM_Tik_adapt(T,b,eta,eta0,u,alpha);
    end
    %volnew(volnew<0) = 0;
    %fprintf('%d-th, alpha = %1.2e,\n',k,alpha/t);
        
    %fprintf('RE of TV adapt is %f.\n',norm(volnew(:)-voltrue(:))/norm(voltrue(:)));

    
    %rhs    = beta * b + v; 
    %volnew = pcg(@(x)AtAReg(x),rhs(:),tol, 100,[],[],vol(:)); 
    %T     = @(x)(beta * CryoEMToeplitzXvol(Heig,x)+ x); %(A'A + alpha*I)*x
    %Minvmat = @(x)(x); 
    %volnew = pcgCryoEM(T,Minvmat, rhs, tol, 100, vol); % A'A x = Atb
    %Costdatafitting(Heig, volnew, rhs);
    %volnew(volnew<0)=0;
    
    %if mod(k,5)==0
    %    tau = LSDF(volnew); 
    %    fprintf('k = %5d  vol error:%f,    tau =%f\n',k,norm(volnew(:)-voltrue(:))/norm(voltrue(:)),tau);
    %end
%     volhat = volnew + volnew - vol;
        
%     dvol   = GradVol3D(volhat); 
%     vxi    = VolGradProjLinfty(vxi - s * dvol);  
   u = wnnm_EM2(volnew,u, sigma*255,voltrue);
    
    %% relative error
    re(k)   = norm(volnew(:)-voltrue(:))/norm(volnew(:));
    
    vol  = volnew;
    %t  = t * 1.05; 
    [MPSNRALLp(k), SSIMALLp(k), FSIMALLp(k)] = quality(real(volnew)*255, double(voltrue)*255);
     fprintf( 'Estimated Image: iter = %2.3f, PSNR = %2.2f \n\n\n', k, MPSNRALLp(k) );
    %if isvar('xtrue'),      ISNR(k) = fun_ISNR(xtrue,g,f);     end
    %iter_time(k) = cputime - start_time;
    %regpar(k)    = mu/t;   
%     ree   = norm(volnew(:,:,35)-vol(:,:,35),'fro')/norm(volnew(:,:,35),'fro');
     Ax = A*volnew; 
     vv = reshape(volnew, [size(volnew,1), size(volnew,2)*size(volnew,3)]);
     tt = svd(vv);
     t = sum(tt);
     fun_val(k) = alpha*norm(Ax(:)-g(:),'fro')^2+t;
end

OutPut.Sol      = vol;
% OutPut.vxi      = vxi;
OutPut.re        = re;
OutPut.psnr        = MPSNRALLp;
OutPut.ssim        = SSIMALLp;
OutPut.fsim        = FSIMALLp;
OutPut.energy        = fun_val;

function q = VolGradProjLinfty(p)
% email: wenyouwei@gmail.com

tmp = sqrt(p(1,:,:,:).^2+p(2,:,:,:).^2+p(3,:,:,:).^2);
tmp(tmp<1) = 1; 

q(1,:,:,:) = p(1,:,:,:)./tmp; 
q(2,:,:,:) = p(2,:,:,:)./tmp; 
q(3,:,:,:) = p(3,:,:,:)./tmp; 


function z = GradVol3D(u)

n  = size(u, 1); 
z  = zeros(3,n, n,n); 
z(1,1:end-1,:,:) = u(2:end,:,:) - u(1:end-1,:,:);  %kernel [0;-1;1]  
z(2,:,1:end-1,:) = u(:,2:end,:) - u(:,1:end-1,:);  %kernel [0 -1 1]
z(3,:,:,1:end-1) = u(:,:,2:end) - u(:,:,1:end-1);    

% z.dx(n,:,:) = zeros(n,n); 
% z.dy(:,n,:) = zeros(n,n);
% z.dz(:,:,n) = zeros(n,n);

function w = Divz3D(z)

zx(:,:,:) = z(1,:,:,:); % x-axis
zy(:,:,:) = z(2,:,:,:); % y-axis
zz(:,:,:) = z(3,:,:,:); % z-axis
n  = size(zx, 1); 

tmp1 = zx(1,:,:); tmp2 = zy(:,1,:); tmp3 = zz(:,:,1); 

tmp1(2:n-1,:,:) = zx(2:n-1,:, :) - zx(1:n-2, :, :);  %kernel [0;-1;1]  
tmp2(:,2:n-1,:) = zy(:,2:n-1, :) - zy(:,1:n-2,  :);  %kernel [0;-1;1]  
tmp3(:,:,2:n-1) = zz(:, :,2:n-1) - zz(:,  :,1:n-2);  

tmp1(n,:,:) = -zx(n-1,:,:); 
tmp2(:,n,:) = -zy(:,n-1,:); 
tmp3(:,:,n) = -zz(:,:,n-1); 

w = tmp1 + tmp2 + tmp3; 
