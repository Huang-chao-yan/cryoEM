function est_rots = ProxGradDescent(C,n_theta,ref_rot)


%% Object:
%  min 1/2|| RiCij - RCji||_2^2
%  s.t  Ri'Ri = I
% Ri = Ri^k - t*\sum (RiCij -RjCji)Cij'
%% Solution:  Proximal Gradient Method (PGM)



TOL=1.0e-14;
K  = size(C,3);
err_c =0;
X = zeros(3,K,K);
%C  = clstack2C( common_lines_matrix,n_theta ); % common line matrixs

for i = 1:K
    for j =1:K
        X(1:2,i,j) = C(1:2,i,j);
    end
end
% X(1:2,K,K) = C;
C = X;

kk =0;
for t = 0.005%:0.005: 0.255
    kk = kk+1;
    
    beta = 1;%0.99;
    %t = 0.618;
    R_old = rand(3,3,K);
    for i = 1:K
        [U,~,V] = svd(R_old(:,:,i));
        R_old(:,:,i) = U * V';
    end
    R_new = R_old;
    
    gradf1 = ObjFunGrad(R_old, C);
    for iter = 1:150
        R_new  = ProxOrth3(R_old - t * gradf1,K);
        [gradf2,Jfun(iter)] = ObjFunGrad(R_new, C);
        
        dR     = R_new - R_old;
        dgradf = gradf2 -gradf1;
        %t      = dR(:)'*dgradf(:)/max(dgradf(:)'*dgradf(:),1e-6); %% BB-step
        %t = 1/(dR(:)'*dR(:)/(dR(:)'*dgradf(:)));
        
        Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));
        if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-2
            break;
        end
        %         t = beta*t;
        gradf1 = gradf2;
        R_old = R_new;
    end
    % figure;
    % plot(1:iter, Errer);title('Error');
    
    %% Make sure that we got rotations.
    est_rots = zeros(3,3,K);
    for k=1:K
        est_rots(:,:,k) = [R_new(:,1:2,k),cross(R_new(:,1,k),R_new(:,2,k))];
        R = est_rots(:,:,k);
        erro = norm(R*R.'-eye(3));
        if erro > TOL || abs(det(R)-1)> TOL
            [U,~,V] = svd(R);
            est_rots(:,:,k) = U*V.';
        end
    end
    norm(est_rots(:)-R_new(:));
    
    MSE(kk) = check_MSE(est_rots, ref_rot);
    plot(Jfun)
    min(Jfun)
    %     inv_est_rot = permute(est_rots, [2 1 3]);
    %     MSEs(kk) = check_MSE(inv_est_rot, ref_rot);
end
%  figure;plot(1:kk,MSE);title('MSE');
% figure;plot(1:kk,MSEs);title('MSEs');



function [gradf, Jfun] = ObjFunGrad(R, C) %LUD
K  = size(C,3);
gradf = zeros(3,3,K);
% for i =1:K
%     Grad_Ri = zeros(3,3);
%     for j = 1:K
%         tmp     = R(:,:,i) *C(:,i,j)-R(:,:,j) *C(:,j,i);
%         tmp     = (tmp/max(norm(tmp,2),1e-4));
%         Grad_Ri = Grad_Ri + tmp * C(:,i,j)';
%     end
%     gradf(:,:,i) = Grad_Ri;
% end

Jfun = 0; 
for i = 1:K    
    A(:,i,:) = R(:,:,i) * squeeze(C(:,i,:)); %
end
B = zeros(size(A)); 
for i = 1:K
    for j = i+1:K
        tmp    = A(:,i,j)- A(:,j,i);
        tmp2   = norm(tmp);  
        Jfun = Jfun + tmp2; 
        tmp    = (tmp/max(tmp2,1e-4));
        B(:,i,j) = tmp; 
        B(:,j,i) = -tmp; 
    end
end
for i = 1:K
    %Grad_Ri = zeros(3,3);
    %for j = 1:K
    %    Grad_Ri = Grad_Ri + B(:,i,j) * C(:,i,j)';
    %end
    %tmp3 = squeeze(B(:,i,:)) * squeeze(C(:,i,:))';
    %norm(tmp3-Grad_Ri)
    %gradf(:,:,i) = Grad_Ri;
    gradf(:,:,i)= squeeze(B(:,i,:)) * squeeze(C(:,i,:))';
end 
% for i = 1:K
%     Grad_Ri = zeros(3,3);
%     for j = 1:K
%         tmp     = A(:,i,j)- A(:,j,i);
%         tmp     = (tmp/max(norm(tmp),1e-4));
%         Grad_Ri = Grad_Ri + tmp * C(:,i,j)';
%     end
%     gradf(:,:,i) = Grad_Ri;
% end


function Rnew = ProxOrth3(R,K)
Rnew = R;
for i =1:K
    tmp = R(:,:,i);
    [U,~,V] = svd(tmp,0);
    Rtmp    = U*V';
    Rtmp    = [Rtmp(:,1:2), cross(Rtmp(:,1), Rtmp(:,2))];
    erro = norm(Rtmp * Rtmp'-eye(3));
    if erro > 1e-10 || abs(det(Rtmp)-1)> 1e-10
        [U,~,V] = svd(Rtmp);
        Rtmp(:,:,k) = U*V';
    end
    Rnew(:,:,i) = Rtmp;
end

