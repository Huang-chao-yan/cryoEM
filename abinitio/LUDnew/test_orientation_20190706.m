K =100;
L =36;
p =1;


%function test_est_orientations(K,L,p)
% K is the number of projections
% L is the number of radial lines within a projection
% p is the probability that correlating two projection images yields the
%    correct common line

%%%%%% Generate K random rotations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                                     %%
% The 3-sphere S^3 in R^4 is a double cover of the rotation group SO(3)
% SO(3) = RP^3
% We identify unit norm quaternions a^2+b^2+c^2+d^2=1 with group elements
% The antipodal points (-a,-b,-c,-d) and (a,b,c,d) are identified as the
% same group elements

rot_matrices = rand_rots(K);

% calculate inverse rotation matrices (just transpose)
inv_rot_matrices = zeros(3,3,K);

inv_rot_matrix = zeros(3);
for k=1:K;
    rot_matrix = rot_matrices(:,:,k);
    inv_rot_matrix = rot_matrix'; % inv(R)=R^T
    inv_rot_matrices(:,:,k) = inv_rot_matrix;
    inv_rot_matrix(:,:,k) = inv(rot_matrices(:,:,k));
end;

n_x(:,:) = inv_rot_matrices(:,1,:);
n_y(:,:) = inv_rot_matrices(:,2,:);
n_z(:,:) = inv_rot_matrices(:,3,:);

fprintf('Computing common lines... ');
% Find common lines -- the intersection points of the circles
common_lines_matrix = zeros(K);

eqs_matrix = zeros(3,4);



fprintf('Finished!\n');

%%%%% Perturb the common lines matrix
%%


%%%%% Test different orientation determination algorithms
MSEs = zeros(6,1);
Time = zeros(6,1);
%% Test est_orientations_LS
tic;
est_inv_rots = est_orientations_LS(common_lines_matrix, L);
Time(1) = toc;
MSEs(1) = check_MSE(est_inv_rots, rot_matrices);

% With spectral norm constraint
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LS(common_lines_matrix, L, pars);
Time(2) = toc;
MSEs(2) = check_MSE(est_inv_rots, rot_matrices);

%% Test est_orientations_LUD
% ADMM
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L);
Time(3) = toc;
MSEs(3) = check_MSE(est_inv_rots, rot_matrices);

% ADMM with spectral norm constraint
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
Time(4) = toc;
MSEs(4) = check_MSE(est_inv_rots, rot_matrices);

% IRLS
pars.solver = 'IRLS';
pars.alpha = 0;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
Time(5) = toc;
MSEs(5) = check_MSE(est_inv_rots, rot_matrices);

% IRLS with spectral norm constraint
pars.alpha = 2/3;
tic;
est_inv_rots = est_orientations_LUD(common_lines_matrix,L, pars);
Time(6) = toc;
MSEs(6) = check_MSE(est_inv_rots, rot_matrices);

%% Print the MSEs and cost time of the results
fprintf('K = %d, L = %d, p = %f\n', K, L, p);
fprintf('=======================================================================================\n')
fprintf(' Exp    LS     LUD     alpha     SDPLR     ADMM      IRLS     MSE             Time\n')
fprintf('  1     Y                          Y                          %1.5f        %6.2f\n', MSEs(1), Time(1));
fprintf('  2     Y               2/3                 Y                 %1.5f        %6.2f\n', MSEs(2), Time(2));
fprintf('  3             Y                           Y                 %1.5f        %6.2f\n', MSEs(3), Time(3));
fprintf('  4             Y       2/3                 Y                 %1.5f        %6.2f\n', MSEs(4), Time(4));
fprintf('  5             Y                                     Y       %1.5f        %6.2f\n', MSEs(5), Time(5));
fprintf('  6             Y       2/3                           Y       %1.5f        %6.2f\n', MSEs(6), Time(6));
fprintf('=======================================================================================\n')

% For example:
% K = 500, L = 360, p = 0.500000
% =======================================================================================
%  Exp    LS     LUD     alpha     SDPLR     ADMM      IRLS     MSE             Time
%   1     Y                          Y                          0.01430          2.26
%   2     Y               2/3                 Y                 0.01584         24.07
%   3             Y                           Y                 0.00150        109.29
%   4             Y       2/3                 Y                 0.00395        138.04
%   5             Y                                     Y       0.00061         32.16
%   6             Y       2/3                           Y       0.00274        1075.34
% =======================================================================================
