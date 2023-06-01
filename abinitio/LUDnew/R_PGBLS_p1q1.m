function est_rots = R_PGBLS_p1q1(C,n_theta, ref_rot)

%% Object:
%  min \sum_{i,j}|| RiCij - RjCji||_1^1 
% L(R,z) = \sum <RiCij - RjCji, z_ij> 
%  s.t   Ri'Ri = I , 
%  s.t   ||z_ij||_{infty} <=1

% Solution:  Proximal Gradient with Backtracking Line-Search
%%

K  = size(C,2);
err_c =0;
TOL = 1e-12;
L = n_theta;

X = zeros(3,K,K);
for i = 1:K
    for j =1:K  
        X(1:2,i,j) = C(1:2,i,j);
    end
end
% X(1:2,K,K) = C;
C = X;

%% test optimal s,t
%kk =0;
% for t = 0.0001:0.0005:0.01
%     kk =kk+1; iter =0; i =0; 
    
s = 0.015;%1.618;
t = 0.018;%1.618;
beta = 0.618;
eta = 0.618;

z_old = rand(3, K, K);
R_old = rand(3,3,K);
for i = 1:K
    [U,~,V] = svd(R_old(:,:,i));
    R_old(:,:,i) = U * V';
end
R_new = R_old;

 for iter = 1:100 
     tk = 1;
     sk = 1;
    flag1 =1;
    flag2 =1;
    kk = 0;
    while flag1
        kk = kk+1;
        % R
        for i =1:K
            grad_Ri = zeros(3,3);
            for j = 1:K
                if j ~= i  
                 grad_Ri = grad_Ri + (z_old(:,i,j) - z_old(:,j,i))*C(:,i,j)'; % 2019-6-26
                end
            end
            tmp = R_old(:,:,i) - tk * grad_Ri;
            [U,~,V] = svd(tmp,0);
             R_new(:,:,i) = U*V';
        end
        
        % Backtracking Line-Search
        [JR,JRK] = convenge_condition1(R_new,R_old,C,z_old,tk);
        if JR > JRK & kk<10
            tk = beta*tk;
            beta = kk/(iter+3);
        else
            flag1 = 0;
        end
    end
     
   kkk = 0;
    while flag2
        kkk = kkk+1;
    % z -------------------------------------------------------
    for i = 1:K
        for j = i + 1:K 
            z_new(:,i,j) = z_old(:,i,j) + sk*(R_new(:,:,i)*C(:,i,j)- R_new(:,:,j)*C(:,j,i));
            z_new(:,j,i) = z_old(:,j,i) + sk*(R_new(:,:,j)*C(:,j,i)- R_new(:,:,i)*C(:,i,j));
            z_new(z_new>1) = 1; z_new(z_new<-1) = -1;    
        end
    end
    
    % Backtracking Line-Search
    [JZ,JZK] = convenge_condition2(R_new,C,z_new,z_old,tk);
        if JZ > JZK & kkk<10
            sk = eta*sk;
            eta = kkk/(kkk+3);
        else
            flag2 = 0;
        end
    end
   Errer(iter) = norm(R_new(:) - R_old(:))/norm(R_new(:));

    % convergence
    if norm(R_new(:) - norm(R_old(:)))/norm(R_new(:)) < 1e-4
        break
    end
    z_old = z_new;
    R_old = R_new;
 end
 
% V1 = squeeze(R_new(:,1,:));
% V2 = squeeze(R_new(:,2,:));
% A = ATA_solver(V1,V2,K);% V1*A'=R1 and V2*A'=R2 
% R1 = A*V1;
% R2 = A*V2;

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
 MSEs = check_MSE(est_rots, ref_rot);
% est_inv_rots = permute(est_rots,[2,1,3]);
% MSEs = check_MSE(est_inv_rots, ref_rot);


figure;plot(1:iter,Errer);



