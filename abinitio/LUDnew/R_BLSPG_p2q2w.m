function est_rots = R_BLSPG_p2q2w(W, C,n_theta, ref_rot)
% 2019-10-14
%% Object:
%  \sum_{ij} \min w_{ij}|| RiCij - RjCji||_2^2
%  = min - \sum_i w_{ij} <Ri, \sum_j (RjCjiCij'>
%  s.t  Ri'Ri = I
% L(R) = - \sum_i  w_{ij} <Ri, \sum_j (RjCjiCij'>
%% Proximal Gradient with Backtracking Line-Search

TOL=1.0e-14;
K  = size(C,3);
err_c =0;
X = zeros(3,K,K);
% C  = clstack2C( common_lines_matrix,n_theta ); % common line matrixs
for i = 1:K
    for j =1:K
        X(1:2,i,j) = C(1:2,i,j);
    end
end
% X(1:2,K,K) = C;
C = X;

tt =0;
for t = 0.5%:0.05: 1.1
    tt = tt+1;
    
    beta = 0.618;%0.99;
    tk = 1;
    R_old = rand(3,3,K);
    for i = 1:K
        [U,~,V] = svd(R_old(:,:,i));
        R_old(:,:,i) = U * V';
    end
    R_new = R_old;
    Z = R_old;
    
    for iter = 1:8
        tk = 1;
        flag =1;
        kk = 0;
        while flag
            % R
            kk = kk+1;
            for i =1:K
                Grad_Ri = zeros(3,3);
                Grad_Zi = zeros(3,3);
                for j = 1:K
                    if i ~=j
                        % Grad_Ri = Grad_Ri + R_old(:,:,j) *C(:,j,i)*C(:,i,j)';
                        Grad_Zi = Grad_Zi + W(i,j)*Z(:,:,j) *C(:,j,i)*C(:,i,j)';
                    end
                end
                temp = Z(:,:,i) - tk * Grad_Zi;
                [U,~,V] = svd(temp,0);
                R_new(:,:,i) = U*V';
            end
            Z = R_new + beta*(R_new - R_old);
            
            [JR,JRK] = convenge_condition (R_new,R_old,C,tk);
            if JR > JRK & kk<30
                beta =(kk)/(kk+3);
                tk = beta*tk;
            else
                flag = 0;
            end
        end
        
        Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));
        %     if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-5
        %         break;
        %     end
        R_old = R_new;
        
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
        
        
        % inv_est_rot = permute(est_rots, [2 1 3]);
        % MSEs(iter) = check_MSE(inv_est_rot, ref_rot);
    end
    figure; plot(1:iter, Errer);title('Error');
    MSE(tt) = check_MSE(est_rots, ref_rot);
end
% figure;plot(1:tt,MSE);title('MSE');
% figure;plot(1:iter,MSEs);title('MSEs');



