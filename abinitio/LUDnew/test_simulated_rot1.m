%% Experiments with simulated rotation  
% 2020-9-11
clc; clear all; close all
%% parameter set
for K = [500, 1000, 2000]
n_theta = 360; %360;%72 120 180
n_r = 100;     %100;    %33

% 1-reference rotation matrix
ref_rot = rand_rots(K);
inv_rot_matrices = permute(ref_rot,[2 1 3]);
% 2-Find common lines -- the intersection points of the circles
fprintf('Computing common lines... ');
% common_lines_matrix= ref_commlines(ref_rot, L);
% Perturb the common lines matrix
is_perturbed = zeros(K); % to verify the success of the algorithm we store which common lines were perturbed

for pp =  [1, 0.8,  0.6,  0.4,  0.2] % the proportion of correctly detected common lines
common_lines_matrix= ref_commlines(ref_rot, n_theta,pp);

%% Test different orientation determination algorithms
% opts.tol = 1e-3;
% pars.alpha = 2/3;
% alpha = pars.alpha;
K = size(common_lines_matrix, 1);
tic;
C = clstack2C( common_lines_matrix,n_theta );
tt = toc;
S = construct_S(common_lines_matrix, n_theta);

%% R: solveing R directly 
% save result
%save CK100SNR32 common_lines_matrix ref_rot p projs
fp = fopen('Result_SimulatedRot1.txt','a+'); 

%% 2- Projection gradient Methods (PGM) for est_rotmatrix
%profile off; profile on; 

% 2.1 PGM for LS model
k = 1;
tic;
est_rots = R_PG_p2q2(C, ref_rot);
Time(k) = toc;
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q2(est_rots(:,1:2,:),C(1:2,:,:));


% 2.2- PGM for LUD model
k = 2; 
tic;
est_rots = R_PG_p2q1(C, ref_rot);
Time(k) = toc; 
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));


%% 3 Manifold Proximal Gradient method (ManPGM)
% 3.1- ManPGM for LS model
k = 3;
Param.MaxIter = 500;
Param.RefRot = ref_rot;
tic;
est_rots = ManProxGradp2q2(C,Param);
Time(k) = toc; 
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));

% 3.2- ManPGM for LUD model
k = 4;
Param.MaxIter = 500;
Param.RefRot = ref_rot;
tic;
est_rots = ManProxGradp2q1(C,Param);
Time(k) = toc; 
[MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));

% 3.3- ManPGM for LUD model based on Adative step
% k = 5;
% Param.MaxIter = 500;
% Param.RefRot = ref_rot;
% tic;
% est_rots = ManPG_Ada_p2q1(C,Param);
% Time(k) = toc 
% [MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
% [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));

%------
fprintf(fp,'----------------------------------\n')
fprintf(fp,' K = %d, L = %d, pp = %.3f\n',  K, n_theta, pp);
fprintf(fp, 'Exp   Method    MSE     Time   \n')
fprintf(fp,'---------------------------------\n')
fprintf(fp,'  1   PGLS     %1.4f  %1.2f \n',   MSEs(1),  Time(1));
fprintf(fp,'  2   PGLUD    %1.4f  %1.2f \n',   MSEs(2),  Time(2));
fprintf(fp,'  3   MPGLS    %1.4f  %1.2f \n',   MSEs(3),  Time(3));
fprintf(fp,'  4   MPGLUD   %1.4f  %1.2f \n',   MSEs(4),  Time(4));
fprintf(fp,'----------------------------------\n');



%% 4 @Wang lanhui and A.mit Singer's Methods 

% 4.1- SDP for LS model
k = 1;
pars.alpha = 0; 
tic;
est_inv_rots = est_orientations_LS(common_lines_matrix, n_theta, pars);
Time(k) = toc;
[MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
[GradRI,ObjFunI(k)] = FunCostAndGradp2q2(est_inv_rots(:,1:2,:),C(1:2,:,:));


% 4.2- ADMMM for LUD model
k=2;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta);
Time(k) = toc;
[MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));

% 4.3- SDP for IRLS model
k= 3;
pars.solver = 'IRLS';
pars.alpha = 0;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
Time(k) = toc;
[MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
[GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));


%% with spectral norm constraint

% 4.4- ADMM for LS model
% k = 4;
% pars.alpha = 2/3; 
% tic;
% est_inv_rots = est_orientations_LS(common_lines_matrix, n_theta, pars,ref_rot);
% Time(k) = toc;
% [MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
% [GradRI,ObjFunI(k)] = FunCostAndGradp2q2(est_inv_rots(:,1:2,:),C(1:2,:,:));


% 4.5- ADMM for LUD model
% k = 5;
% pars.alpha = 2/3;
% pars.solver = 'ADMM';
% tic;
% est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
% Time(k) = toc;
% [MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
% [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));


% 4.6- ADMM for IRLS model
% k= 6;
% pars.solver = 'IRLS';
% pars.alpha = 2/3;
% tic;
% est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
% Time(k) = toc;
% [MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
% [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));



%% Print the MSEs and cost time of the results
fprintf(fp,'K = %d, pp = %.3f \n', K, pp);
fprintf(fp, 'Exp  Method   MSE   Time     \n');
fprintf(fp,'-----------------------------------------\n')
fprintf(fp,'  1   LS    %1.4f  %1.2f  \n',   MSEs(1),  Time(1));
fprintf(fp,'  2   LUD   %1.4f  %1.2f  \n',   MSEs(2),  Time(2));
fprintf(fp,'  3  IRLS   %1.4f  %1.2f  \n',   MSEs(3),  Time(3));
% fprintf(fp,'  4   LSc   %1.4f  %1.2f  \n',   MSEs(4),  Time(4));
% fprintf(fp,'  5   LUDc  %1.4f  %1.2f  \n',   MSEs(5),  Time(5));
% fprintf(fp,'  6  IRLSc  %1.4f  %1.2f  \n',   MSEs(6),  Time(6));
fprintf(fp,'-------------------------------------------\n')

end

end
