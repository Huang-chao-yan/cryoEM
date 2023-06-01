function OutPut = CryoEM_TF_admm(A,g,T,b,eta,eta0,sigma,voltrue)


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
p = vol;
% vxi    = GradVol3D(vol);
%beta   = t * alpha; 
alpha  = 0.05;%1e-1; 

while (k<MaxIter)%&&(re(k)>SolRE)
    k = k+1;  
    
    %% update the primal variable 
%     v      = vol - t * Divz3D(vxi);  %v=vk-t div eta k; section 3.3.1 of the corresponding paper (HCY 211116)
    u = u - p./(2*eta);
    
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
   
   volnew = volnew+p./(2*eta);
   u = wnnm_EM2(volnew,u, sigma*255,voltrue);
    
   %p multi
   p = p + t*(volnew - u);
    
   
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
     t1 = sum(tt);
     fun_val(k) = alpha*norm(Ax(:)-g(:),'fro')^2+t1;
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



