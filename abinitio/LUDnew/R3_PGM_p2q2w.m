function est_rots = R3_PGM_p2q2w(C, n_theta, ref_rot)
% 2019-10-14
%% Object:
%  \sum_{ij} \min w_{ij}|| RiCij - RjCji||_2^2
%  s.t  Ri'Ri = I
%  Ri = Ri^k - w_{ij}*t*\sum (RiCij -RjCji)Cij' 
%  w_{ij} = 1/ sqrt(|| RiCij - RjCji||_2^2)
% Solution: Projected gradient algorithm  (PGM) 
%%
epsilon = 1e-8;
TOL=1.0e-14;
K  = size(C,3);
Wij = 1;

err_c =0;
X = zeros(3,K,K);
for i = 1:K
    for j =1:K
        X(1:2,i,j) = C(1:2,i,j);
    end
end
C = X;

kk =0; 
 for t = 0.0059% :0.00001: 0.006    % 0.015
    kk = kk+1;
    beta = 1;%0.99;
    %t = 0.618;
    z_old = rand(3, K, K);
    R_old = rand(3,3,K);
    for i = 1:K
        [U,~,V] = svd(R_old(:,:,i));
        R_old(:,:,i) = U * V';
    end
    R_new = R_old;
        
    for iter = 1:100
        for i =1:K
            Grad_Ri = zeros(3,3);
            for j = 1:K
                % W
                wij = 2 - 2*(R_old(:,:,i)*C(:,i,j))'*(R_old(:,:,j)*C(:,j,i));
%                 wij = norm(R_old(:,:,i)*C(:,i,j)-R_old(:,:,j)*C(:,j,i),2)^2;
                Wij = 1/sqrt(wij + epsilon^2);
                % Grad_Ri
                Grad_Ri = Grad_Ri + Wij* (R_old(:,:,i) *C(:,i,j)-R_old(:,:,j) *C(:,j,i))*C(:,i,j)';
            end
            tmp = R_old(:,:,i) - t* Grad_Ri;
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
    
    MSE1 = check_MSE(est_rots, ref_rot);
    MSE2 = check_MSE(R_new, ref_rot);
    [MSE1,MSE2]

  
    MSE(kk) = check_MSE(est_rots, ref_rot);

end
  figure;plot(1:kk,MSE);title('MSE');
% figure;plot(1:kk,MSEs);title('MSEs');




