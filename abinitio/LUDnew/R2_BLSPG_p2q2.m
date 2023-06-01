function est_rots = R2_BLSPG_p2q2(C, ref_rot,R_old)
% 2019-10-14
%% Object:
%  \sum_{ij} \min 1/2*|| RiCij - RjCji||_2^2
%  s.t  Ri'Ri = I
%  Ri^{k+1} = Ri - t*(RiCij - RjCji)*Cij^{T}
%% Proximal Gradient with Backtracking Line-Search

TOL=1.0e-14;
K  = size(C,3);
err_c =0;
X = zeros(3,K,K);
X(1:2,:,:) = C(1:2,:,:);
C = X;

tt =0;
for t0 = 1%:0.05: 1
    t= 0.05;%0.618;%0.015
    tt = tt+1;
    alpha = 0.618;%0.99;
    if ~isvar('R_old')
        R_old   = rand(3,3,K);
        for i = 1:K
            [U,~,V]      = svd(R_old(:,:,i));
            R_old(:,:,i) = U * V';
        end
    end
    R_new = R_old;
    Z = R_old;
    
    for iter = 1:40
        flag =1;
        kk = 0;
        while flag
            % R
            kk = kk+1;
            for i =1:K
                Grad_Ri = zeros(3,3);
                %                 Grad_Zi = zeros(3,3);
                for j = 1:K
                    Grad_Ri = Grad_Ri + (R_old(:,:,i) *C(:,i,j)-R_old(:,:,j) *C(:,j,i))*C(:,i,j)';
                    % Grad_Zi = Grad_Zi + Z(:,:,j) *C(:,j,i)*C(:,i,j)';
                end
                temp = R_old(:,:,i) - t * Grad_Ri;
                [U,~,V] = svd(temp,0);
                R_new(:,:,i) = U*V';
            end
            % Z = R_new + beta*(R_new - R_old);
            
            [JR,JRK] = convenge_condition (R_new,R_old,C,t);
            if JR > JRK && kk<10
                t = alpha*t0;
            else
                flag = 0;
            end
        end
        Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));
        
        %     if norm(R_new(:) - R_old(:))/norm(R_new(:))<1e-5
        %         break;
        %     end
        R_old = R_new;
        t0 = t;
    end
    
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
    %     figure; plot(1:iter, Errer);title('Error');
    MSE(tt) = check_MSE(est_rots, ref_rot);
end
% figure;plot(1:tt,MSE);title('MSE');
% figure;plot(1:iter,MSEs);title('MSEs');



