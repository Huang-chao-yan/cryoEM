function est_rots = R2_PEG_p2q2(C,ref_rot,R_old)


%% Object:
%  min 1/2|| RiCij - RCji||_2^2
%  s.t  Ri'Ri = I
% Ri = Ri^k - t*\sum (RiCij -RjCji)Cij'
%% Solution:  Proximal Gradient Method (PGM)

TOL=1.0e-14;
K  = size(C,3);
err_c =0;
X = zeros(3,K,K);
X(1:2,:,:) = C;
C = X;

kk =0;
for t = 0.5%:0.005: 0.255
    kk = kk+1;
    
    beta = 1;%0.99;
    %t = 0.618;
  if ~exist('R_old','var')
    R_old   = rand(3,3,K);
    for i = 1:K
        [U,~,V]      = svd(R_old(:,:,i));
        R_old(:,:,i) = U * V';
    end
end

    for iter = 1:100
        for i =1:K
            Grad_Ri = zeros(3,3);
            for j = 1:K
                Grad_Ri = Grad_Ri +(R_old(:,:,i) *C(:,i,j)-R_old(:,:,j) *C(:,j,i))*C(:,i,j)';
            end
            tmp = R_old(:,:,i) - t * Grad_Ri;
            [U,~,V] = svd(tmp,0);
            R_new(:,:,i) = U*V';
        end
        
        Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));
        if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-2
            break;
        end
%         t = beta*t;
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
%     inv_est_rot = permute(est_rots, [2 1 3]);
%     MSEs(kk) = check_MSE(inv_est_rot, ref_rot);
 end
% figure;plot(1:kk,MSE);title('MSE');
% figure;plot(1:kk,MSEs);title('MSEs');





