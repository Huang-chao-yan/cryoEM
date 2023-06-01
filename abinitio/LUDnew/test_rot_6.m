
clc; clear all; close all

K =100;
L =100;
n_theta = L;

ref_rot = rand_rots(K);
inv_rot_matrices = permute(ref_rot,[2 1 3]);

% Find common lines -- the intersection points of the circles
fprintf('Computing common lines... ');
% common_lines_matrix= ref_commlines(ref_rot, L);

% Perturb the common lines matrix
is_perturbed = zeros(K); % to verify the success of the algorithm we store which common lines were perturbed
pp = 0 ;
common_lines_matrix= ref_commlines(ref_rot, n_theta,pp);

%% Test different orientation determination algorithms

opts.tol = 1e-3;
pars.alpha = 2/3;
alpha = pars.alpha;
K = size(common_lines_matrix, 1);

C = clstack2C( common_lines_matrix,n_theta );
S = construct_S(common_lines_matrix, n_theta);

%% R: solveing R directly 

% initial value 
R_old = rand(3,3,K);
for i = 1:K
    [U,~,V] = svd(R_old(:,:,i));
    R_old(:,:,i) = U * V';
end


% Model p = 2, q =2 
k = 1;
tic;
est_rots = R2_PEG_p2q2(C,ref_rot,R_old);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);


k = 6;
tic;
est_rots = EstimateRotateMat(C,ref_rot,R_old);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 2;
tic;
est_rots = R2_BLSPG_p2q2(C, ref_rot,R_old);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 3;
W = ones(2*K);
tic;
for j = 1:11
    est_rots = R2_PGM_p2q2w(W, C,n_theta, ref_rot);
    [W, res] = W_weights(est_rots, C);
     MSEss(j) = check_MSE(est_rots, ref_rot);
end
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);
figure; plot(1:j,MSEss);title('MSEs');

k = 4;
tic;
est_rots = R3_PGM_p2q2w(C, n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 5;
tic;
est_rots = R2_PEG_p1q1(C,n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot); 

 
fprintf('K = %d, L = %d, pp = %1.1f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method         MSE            Time\n')
fprintf('  1   R2_PEG_p2q2     %1.5f     %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   R2_BLSPG_p2q2   %1.5f     %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   R2_PGM_p2q2w    %1.5f     %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   R3_PGM_p2q2w    %1.5f     %6.2f\n',   MSE(4),  Time(4));
fprintf('  5   R2_PEG_p1q1     %1.5f     %6.2f\n',   MSE(5),  Time(5));
fprintf('-------------------------------------------\n');


%% Model p = 2, q = 2;
k = 1;
tic;
est_rots = R_PGM_p2q2(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 2;
tic;
est_rots = R_APGM_p2q2(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k=3;
tic;
est_rots = R_BLSPG_p2q2(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 4;
tic;
est_rots = R_PEGM_p2q2(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   R_PGM_p2q2     %1.5f      %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   R_APGM_p2q2    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   R_BLSPG_p2q2   %1.5f      %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   R_PEGM_p2q2    %1.5f      %6.2f\n',   MSE(4),  Time(4));
fprintf('-------------------------------------------\n')

%%  Model p = 2, q = 1/2

k = 1;
W = ones(2*K);
tic;
for j = 1:5
    est_rots = R_PGM_p2q2w(W, C,n_theta, ref_rot);
    [W, res] = W_weights(est_rots, C);
end
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 2;
W = ones(2*K);
tic;
for j = 1:5
    est_rots = R_APGM_p2q2w(W,C,n_theta, ref_rot);
    [W, res] = W_weights(est_rots, C);
end
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k=3;
W = ones(2*K);
tic;
for j = 1:5
    est_rots = R_BLSPG_p2q2w(W, C,n_theta, ref_rot);
    [W, res] = W_weights(est_rots, C);
end
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

k = 4;
W = ones(2*K);
tic;
W = ones(2*K);
tic;
for j = 1:5
    est_rots = R_PEGM_p2q2w(W,C,n_theta, ref_rot);
    [W, res] = W_weights(est_rots, C);
end
Time(k) = toc;
MSE(k) = check_MSE(est_rots, ref_rot);

%%  Model p = 2, q = 1/2
fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   R_PGM_p2q2w     %1.5f      %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   R_APGM_p2q2w    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   R_BLSPG_p2q2w   %1.5f      %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   R_PEGM_p2q2w    %1.5f      %6.2f\n',   MSE(4),  Time(4));
fprintf('-------------------------------------------\n')


%% Model p = 2, q= 1;

k = 1;
tic;
est_inv_rots = R_PDM_p2q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)
% inv_est_rot = permute(est_inv_rots, [2 1 3]);
% MSEs(k) = check_MSE(est_inv_rots, ref_rot);

k = 2;
tic;
est_inv_rots = R_PEGM_p2q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
inv_est_rot = permute(est_inv_rots, [2 1 3]);
MSEs(k) = check_MSE(est_inv_rots, ref_rot);

k = 3;
tic;
est_inv_rots = R_PGBLS_p2q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
inv_est_rot = permute(est_inv_rots, [2 1 3]);
MSEs(k) = check_MSE(est_inv_rots, ref_rot);


fprintf('K = %d, L = %d, pp= %f \n', K, n_theta,pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   R_PDM_p2q1     %1.5f      %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   R_PEGM_p2q1    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   R_PGBLS_p2q1   %1.5f      %6.2f\n',   MSE(3),  Time(3));
fprintf('-------------------------------------------\n')


%% Model p = 1, q= 1;

k = 1;
tic;
est_inv_rots = R_PDM_p1q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
% est_inv_rots = permute(est_rots,[2,1,3]);
% MSEs = check_MSE(est_inv_rots, ref_rot);

k = 2;
tic;
est_inv_rots = R_PEGM_p1q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);

k = 3;
tic;
est_inv_rots = R_PGBLS_p1q1(C,n_theta, ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);

fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   R_PDM_p1q1     %1.5f      %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   R_PEGM_p1q1    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   R_PGBLS_p1q1   %1.5f      %6.2f\n',   MSE(3),  Time(3));
fprintf('-------------------------------------------\n')




%% G  -----------------------------------------------------

%% Model p = 2, q= 2;  with NO spectral constrain

k = 1;
tic;
est_inv_rots = G_PDM_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)
k = 2;
tic;
est_inv_rots = G_PEGM_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)
k = 3;
tic
est_inv_rots = G_APGM_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)

k = 4;
tic;est_inv_rots = G_PDMX_p2q2(S, n_theta,ref_rot);

Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)
k = 5;
tic;
est_inv_rots = G_PEGMX_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)

k = 6;
tic;
est_inv_rots = G_APGMX_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)


%% Model p = 2, q= 2; with spectral constrain
k = 7;
tic;
est_inv_rots = G_PDMC_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
MSE(k)

k = 8;
tic;
est_inv_rots = G_PEGMC_p2q2(S, n_theta,ref_rot);
Time(k) = toc;
MSE(8) = check_MSE(est_inv_rots, ref_rot);
MSE(8)


% fprintf('SNR = %f, p = %f\n',  SNR, p);
fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method         MSE            Time\n')
fprintf('  1   G_PDM_p2q2     %1.5f     %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   G_PEGM_p2q2    %1.5f     %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   G_APGM_p2q2    %1.5f     %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   G_PDMX_p2q2    %1.5f     %6.2f\n',   MSE(4),  Time(4));
fprintf('  5   G_PEGMX_p2q2   %1.5f     %6.2f\n',   MSE(5),  Time(5));
fprintf('  6   G_APGMX_p2q2   %1.5f     %6.2f\n',   MSE(6),  Time(6));
fprintf('  7   G_PDMC_p2q2    %1.5f     %6.2f\n',   MSE(7),  Time(7));
fprintf('  8   G_PEGMC_p2q2   %1.5f     %6.2f\n',   MSE(8),  Time(8));
fprintf('-------------------------------------------\n')



%% Model p = 2, q= 1; with NO spectral constrain
k = 1;
tic;
est_inv_rots = G_PDM_p2q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 2;
tic;
est_inv_rots = G_PEGM_p2q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 3;
tic;
est_inv_rots = G_PDMX_p2q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 4;
tic;
est_inv_rots = G_PEGMX_p2q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);


%% Model p = 2, q= 1; with spectral constrain
k = 5;
tic;
est_inv_rots = G_PDMC_p2q1(C, n_theta,alpha*K,opts,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 6;
tic;
est_inv_rots = G_PEGMC_p2q1(C, n_theta,alpha*K,opts,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);



% fprintf('SNR = %f, p = %f\n',  SNR, p);
fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method         MSE            Time\n')
fprintf('  1   G_PDM_p2q1     %1.5f     %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   G_PEGM_p2q1    %1.5f     %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   G_PDMX_p2q1    %1.5f     %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   G_PEGMX_p2q1   %1.5f     %6.2f\n',   MSE(4),  Time(4));
fprintf('  5   G_PDMC_p2q1    %1.5f     %6.2f\n',   MSE(5),  Time(5));
fprintf('  6   G_PEGMC_p2q1   %1.5f     %6.2f\n',   MSE(6),  Time(6));
fprintf('-------------------------------------------\n')


%% Model p = 1, q= 1; with NO spectral constrain

k = 1;
tic;
est_inv_rots = G_PDM_p1q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 2;
est_inv_rots = G_PEGM_p1q1(C, n_theta,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);


%% Model p = 1, q= 1; with spectral constrain
k = 3;
tic;
est_inv_rots = G_PDMC_p1q1(C, n_theta,alpha*K,opts,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);
k = 4;
tic;
est_inv_rots = G_PEGMC_p1q1(C, n_theta,alpha*K,opts,ref_rot);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot) ;

% fprintf('SNR = %f, p = %f\n',  SNR, p);
fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method         MSE            Time\n')
fprintf('  1   G_PDM_p1q1     %1.5f     %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   G_PEGMX_p1q1   %1.5f     %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   G_PDMC_p1q1    %1.5f     %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   G_PEGMC_p1q1   %1.5f     %6.2f\n',   MSE(4),  Time(4));
fprintf('-------------------------------------------\n')


%% Model p = 2, q= 1/2;  with NO spectral constrain
k = 1;
tic;
pars.solver = 'G_PDM_p2q2w';
est_inv_rots = G_PDM_p2q2W( S, n_theta,ref_rot,pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot) ;
k = 2;
tic;
pars.solver = 'G_PEGM_p2q2w';
est_inv_rots = G_PDM_p2q2W( S, n_theta,ref_rot,pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot) ;


%% Model p = 2, q= 1/2;  with spectral constrain
k = 3;
tic;
pars.solver = 'G_PDMC_p2q2w';
est_inv_rots = G_PDM_p2q2W( S, n_theta,ref_rot,pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot) ;

k = 4;
tic;
pars.solver = 'G_PEGMC_p2q2w';
est_inv_rots = G_PDM_p2q2W( S, n_theta,ref_rot,pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot) ;


% fprintf('SNR = %f, p = %f\n',  SNR, p);
fprintf('K = %d, L = %d, pp = %f\n', K, n_theta, pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   G_PDM_p2q2w     %1.5f      %6.2f\n',   MSE(1),  Time(1));
fprintf('  2   G_PEGM_p2q2w    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  3   G_PDMC_p2q2w    %1.5f      %6.2f\n',   MSE(3),  Time(3));
fprintf('  4   G_PEGMC_p2q2w   %1.5f      %6.2f\n',   MSE(4),  Time(4));
fprintf('-------------------------------------------\n')




%%  wang lan hui code
%%
% Test est_orientations_LS
% k =1
% tic;
% est_inv_rots = est_orientations_LS(common_lines_matrix, L);
% Time(k) = toc;
% MSEs(k) = check_MSE(est_inv_rots, ref_rot);

% With spectral norm constraint
k =2;
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LS(common_lines_matrix, L, pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);

%% Test est_orientations_LUD
% ADMM
k=3;
pars.solver = 'ADMM';
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);

% ADMM with spectral norm constraint
k=4;
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);

% IRLS
% pars.solver = 'IRLS';
% pars.alpha = 0;
% tic;
% est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
% Time(5) = toc;
% MSEs(17) = check_MSE(est_inv_rots, rot_matrices);

% IRLS with spectral norm constraint
k=5;
pars.solver = 'IRLS';
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
Time(k) = toc;
MSE(k) = check_MSE(est_inv_rots, ref_rot);


fprintf('K = %d, L = %d , pp =%f\n', K, n_theta,pp);
fprintf('-----------------------------------------\n')
fprintf(' Exp  Method          MSE         Time\n')
fprintf('  1   G_ADMMC_p2q2    %1.5f      %6.2f\n',   MSE(2),  Time(2));
fprintf('  2   G_ADMM_p2q1     %1.5f      %6.2f\n',    MSE(3),  Time(3));
fprintf('  3   G_ADMMC_p2q2w   %1.5f      %6.2f\n',   MSE(4),  Time(4));
fprintf('  4   G_IRLSC_p2q2w   %1.5f      %6.2f\n',   MSE(5),  Time(5));
fprintf('-------------------------------------------\n')





