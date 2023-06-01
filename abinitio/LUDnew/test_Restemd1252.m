%% Experiments with simulated noisy projection
clear; clc; close all


%% Generate simulated projections
% Generate 200 simulated projections of size 65x65.
% For simplicity, the projections are centered.
%  load cleanrib;
 %volref = ReadMRC('emd_1252.map');
  volref = ReadMRC('emd_1252.map');
   fname = 'emd1252';
figure;isosurface(real(volref),max(real(volref(:)))/5);axis off;%axis
% vis3d tight;
% title('OR');
  print('-djpeg',strcat('./figRestemd1252/',fname,'ORG.jpg'));
 
% print('-djpeg',strcat('/home/panh/panhuan/aspirev013org/aspirev013org/abinitio/C1/LUD_new/figsph/vol',fname,'ORG.jpg'));
fp = fopen('Result_emd1252Rest.txt','a+');


n = size(volref,1);
% n = ;  %65
% V1 = zeros(n,n,n);
% V1(2:k+1,2:k+1,2:k+1)= volref;
% volref = V1;


n1 = 137;  % n = 65;129 137%
V = NewSizeVol(volref,n1);
volref1 = V;

for K =  500
for SNR =[8,4,2,1, 1/2, 1/4]
    %SNR=1000; % No noise
    %[projs,noisy_projs,~,ref_rot]=cryo_gen_projections(n,K,SNR);
    
    a          = qrand(K);    %
    ref_rot    = q_to_rot(a); %quat2rotm(a);
    A     = OpNufft3D(ref_rot,n); % projection operatorz
    projs = A * volref;
    
    % figure;viewstack(projs,5,5); % Show some noisy projections
    [noisy_projs, sigma] = ProjAddNoise(projs, SNR);
        upbound = sum(sigma(:).^2)*n*n;     % eta: the sum of the variance of the noise
        eta0    = sum(noisy_projs(:).*conj(noisy_projs(:)));
%         erra = norm(noisy_projs(:)-projs(:))^2;
%         err_b = abs(erra-upbound)/norm(noisy_projs(:));
%     
    % figure;viewstack(noisy_projs,5,5); % Show some noisy projections
 %%    
    
    A1     = OpNufft3D(ref_rot,n1);
    projs1 = A1 * volref1;
    [noisy_projs1, sigma1] = ProjAddNoise(projs1, SNR);
    masked_r = 45;
    masked_projs=mask_fuzzy(noisy_projs1,masked_r); % Applly circular mask
    % figure;viewstack(masked_projs,5,5); % Show some noisy projections
    % Compute polar Fourier transform, using radial resolution n_r and angular
    % resolution n_theta. n_theta is the same as above.
    n_theta = 72; %360;%72
    n_r = n;     %100;    %33
    [npf,sampling_freqs]=cryo_pft(masked_projs,n_r,n_theta,'single');  % take Fourier transform of projections
    
    % Find common lines from projections
    max_shift=0;
    shift_step=1;
    common_lines_matrix = commonlines_gaussian(npf,max_shift,shift_step);
    C = clstack2C( common_lines_matrix,n_theta );
    % Find reference common lines and compare
    % [ref_clstack,~]=clmatrix_cheat_qq(ref_rot,n_theta);
    [ref_clstack,~]=clmatrix_cheat(ref_rot,n_theta);
    % cheack common lines accurate rate
    p = comparecl( common_lines_matrix, ref_clstack, n_theta, 10 );
    fprintf('Percentage of correct common lines: %f%%\n\n',p*100);
    
    %% 2- Projection gradient Methods(PGM) for est_rotmatrix
    %profile off; profile on;
    
    % 2.1 PGM for LS model
    k = 1;
    tic;
    est_rots = R_PG_p2q2(C, ref_rot);
    Time(k) = toc;
    [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q2(est_rots(:,1:2,:),C(1:2,:,:));
    [MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
   
    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x   
    V_p2q2 = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    fprintf(fp,'The RE of PGLS is %f.\n',norm(V_p2q2(:)-volref(:))/norm(volref(:)));
    Err_v(k) = norm(volref(:) - real(V_p2q2(:)))/norm(volref(:));
    figure;isosurface(real(V_p2q2),max(real(V_p2q2(:)))/5);axis off;%title('R2BLSPGp2q2');
     print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'PGMp2q2.jpg'));
    
    
    % 2.2- PGM for LUD model
    k = 2;
    tic;
    est_rots = R_PG_p2q1(C, ref_rot);
    Time(k) = toc;
    [MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
    [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));
    
    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x
       
    V_p2q1 = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(V_p2q1(:)))/norm(volref(:));
    figure;isosurface(real(V_p2q1),max(real(V_p2q1(:)))/5);axis off;%title('R2BLSPGp2q2');%axis vis3d tight;
     print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'PGMp2q1.jpg'));
    
    
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
    
    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x    
    V_Mp2q2 = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(V_Mp2q2(:)))/norm(volref(:));
    figure;isosurface(real(V_Mp2q2),max(real(V_Mp2q2(:)))/5);axis off;%title('R2BLSPGp2q2');
    %axis vis3d tight;
    print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'MPGMp2q2.jpg'));
    
    % 3.2- ManPGM for LUD model
    k = 4;
    Param.MaxIter = 500;
    Param.RefRot = ref_rot;
    tic;
    est_rots = ManProxGradp2q1(C,Param);
    Time(k) = toc;
    [MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
    [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));
    
    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x
       
    V_Mp2q1 = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(V_Mp2q1(:)))/norm(volref(:));
    figure;isosurface(real(V_Mp2q1),max(real(V_Mp2q1(:)))/5);axis off;%title('R2BLSPGp2q2');
    axis vis3d tight;
    print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'MPGMp2q1.jpg'));
    
    % 3.3- ManPGM for LUD model based on Adative step
    % k = 5;
    % Param.MaxIter = 500;
    % Param.RefRot = ref_rot;
    % tic;
    % est_rots = ManPG_Ada_p2q1(C,Param);
    % Time(k) = toc
    % [MSEs(k),est_inv_rots, err]= check_MSE(est_rots, ref_rot);
    % [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_rots(:,1:2,:),C(1:2,:,:));
    % fprintf('Reconstruction projections  from clean centered projections \n');
    % [ V_MAp2q1, v_b, kernel, err, iter, flag] = recon3d_firm(projs,est_inv_rots,[], 1e-4, 500, zeros(n,n,n));
    % Err_v(k) = norm(volref(:) - real(V_MAp2q1(:)))/norm(volref(:));
    % figure;isosurface(real(V_MAp2q1),max(real(V_MAp2q1(:)))/5);axis off;%title('R2BLSPGp2q2');
    % axis off;%axis vis3d tight;
    % print('-djpeg',strcat('./figsph/vol',fname,num2str(K),num2str(1./SNR),'MAPGMp2q1.jpg'));
    
    
    fprintf(fp,' -----------------------------------------\n')
    fprintf(fp,'SNR = %.3f, K = %d, L = %d, pp = %.3f\n', SNR, K, n_theta, p);
    fprintf(fp, 'Exp   Method      MSE     Time   Err_v    Obj \n')
    fprintf(fp,'-------------------------------------------------\n')
    fprintf(fp,'  1   R_PG_p2q2   %1.4f  %6.2f  %6.3f  %.1f\n',   MSEs(1),  Time(1),  MSEs(1), ObjFunI(1));
    fprintf(fp,'  2   R_PG_p2q1   %1.4f  %6.2f  %6.3f  %.1f\n',   MSEs(2),  Time(2),  MSEs(1), ObjFunI(2));
    fprintf(fp,'  3   R_MPG_p2q2  %1.4f  %6.2f  %6.3f  %.1f\n',   MSEs(3),  Time(3),  MSEs(1), ObjFunI(3));
    fprintf(fp,'  4   R_MPG_p2q1  %1.4f  %6.2f  %6.3f  %.1f\n',   MSEs(4),  Time(4),  MSEs(1), ObjFunI(4));
    %fprintf(fp,'  4   R2_MPGA_p2q1   %1.4f  %6.2f  %6.3f  %.1f\n',   MSEs(4),  Time(4), ObjFunI(1), ObjFunI(4));
    fprintf(fp,'-------------------------------------------------\n')
    
    
    %% 4 @Wang lanhui and A.mit Singer's Methods
    
    % 4.1- SDP for LS model
    k = 1;
    pars.alpha = 0;
    tic;
    est_inv_rots = est_orientations_LS(common_lines_matrix, n_theta);
    Time(k) = toc;
    [MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
    [GradRI,ObjFunI(k)] = FunCostAndGradp2q2(est_inv_rots(:,1:2,:),C(1:2,:,:));

    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x 
    v_LS = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(v_LS(:)))/norm(volref(:));
    figure;isosurface(real(v_LS),max(real(v_LS(:)))/5);axis off;%axis vis3d tight;%title('LSc');
    print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'LS.jpg'));
    
    
    % 4.2- ADMMM for LUD model
    k=2;
    tic;
    est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta);
    Time(k) = toc;
    [MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
    [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));

    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x 
    v_LUDADMM = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(v_LUDADMM(:)))/norm(volref(:));
    figure;isosurface(real(v_LUDADMM),max(real(v_LUDADMM(:)))/5);axis off;%axis vis3d tight;title('ADMM');
    print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'LUD.jpg'));
    
    
    % 4.3- SDP for IRLS model
    k= 3;
    pars.solver = 'IRLS';
    tic;
    est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
    Time(k) = toc;
    [MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
    [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));

    rotmatricesinv = permute(est_inv_rots, [2 1 3]);
    A     = OpNufft3D(rotmatricesinv,n);%% back projection operator
    back_projs = A' * noisy_projs;  % back projection
    Toepker    = compkernel(A);
    ToepCirEig = CryoEMToeplitz2CirEig(Toepker); % compute kernel eigenvalue of Toeplitz matrix
    T       = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A)*x 
    v_IRLS = CryoEM_Tik_adapt(T,back_projs,upbound,eta0);
    Err_v(k) = norm(volref(:) - real(v_IRLS(:)))/norm(volref(:));
    figure;isosurface(real(v_IRLS),max(real(v_IRLS(:)))/5);axis off;%axis vis3d tight;%title('IRLSc');
    print('-djpeg',strcat('./figRestemd1252/',fname,num2str(K),num2str(1./SNR),'IRLS.jpg'));
    
    
%     %% with spectral norm constraint
%     
%     % 4.4- ADMM for LS model
%     k = 4;
%     pars.alpha = 2/3;
%     tic;
%     est_inv_rots = est_orientations_LS(common_lines_matrix, n_theta, pars);
%     Time(k) = toc;
%     [MSEs(k), est_inv_rots]= check_MSE(est_inv_rots, ref_rot);
%     [GradRI,ObjFunI(k)] = FunCostAndGradp2q2(est_inv_rots(:,1:2,:),C(1:2,:,:));
%     [ v_LSc, v_b, kernel ,err, iter, flag] = recon3d_firm(noisy_projs,est_inv_rots,[], 1e-4, 500, zeros(n,n,n));
%     Err_v(k) = norm(volref(:) - real(v_LSc(:)))/norm(volref(:));
%     figure;isosurface(real(v_LSc),max(real(v_LSc(:)))/5);axis off;%axis vis3d tight;%title('LSc');
%     print('-djpeg',strcat('./figsph/vol',fname,num2str(K),num2str(1./SNR),'LSc.jpg'));
%     
%     
%     % 4.5- ADMM for LUD model
%     k = 5;
%     pars.alpha = 2/3;
%     pars.solver = 'ADMM';
%     tic;
%     est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
%     Time(k) = toc;
%     [MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
%     [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));
%     [ v_LUDADMMc, v_b, kernel ,err, iter, flag] = recon3d_firm(noisy_projs,est_inv_rots,[], 1e-4, 500, zeros(n,n,n));
%     Err_v(k)= norm(volref(:) - real(v_LUDADMMc(:)))/norm(volref(:));
%     figure;isosurface(real(v_LUDADMMc),max(real(v_LUDADMMc(:)))/5);axis off;%axis vis3d tight;
%     %title('ADMMc');
%     print('-djpeg',strcat('./figsph/vol',fname,num2str(K),num2str(1./SNR),'LUDc.jpg'));
%     
%     
%     % 4.6- ADMM for IRLS model
%     k= 6;
%     pars.solver = 'IRLS';
%     pars.alpha = 2/3;
%     tic;
%     est_inv_rots = est_orientations_LUD(common_lines_matrix,n_theta, pars);
%     Time(k) = toc;
%     [MSEs(k), est_inv_rots] = check_MSE(est_inv_rots, ref_rot);
%     [GradRI,ObjFunI(k), BI] = FunCostAndGradp2q1(est_inv_rots(:,1:2,:),C(1:2,:,:));
%     [ v_IRLSc, v_b, kernel ,err, iter, flag] = recon3d_firm(noisy_projs,est_inv_rots,[], 1e-4, 500, zeros(n,n,n));
%     Err_v(k) = norm(volref(:) - real(v_IRLSc(:)))/norm(volref(:));
%     figure;isosurface(real(v_IRLSc),max(real(v_IRLSc(:)))/6);axis off;%axis vis3d tight;%title('IRLSc');
%     print('-djpeg',strcat('./figsph/vol',fname,num2str(K),num2str(1./SNR),'IRLSc.jpg'));
%     
    
    %% Print the MSEs and cost time of the results
    fprintf(fp,'SNR = %.3f,K = %d,masked_r = %d, pp = %.3f \n',SNR, K,masked_r, p);
    fprintf(fp, 'Exp  Method   MSE   Time     Err_v    Obj \n');
    fprintf(fp,'-----------------------------------------\n')
    fprintf(fp,'  1    LS  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(1),  Time(1),MSEs(1),ObjFunI(1));
    fprintf(fp,'  2   LUD  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(2),  Time(2),MSEs(1),ObjFunI(2));
    fprintf(fp,'  3  IRLS  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(3),  Time(3),MSEs(1),ObjFunI(3));
%     fprintf(fp,'  4   LSc  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(4),  Time(4),Err_v(4),ObjFunI(4));
%     fprintf(fp,'  5  LUDc  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(5),  Time(5),Err_v(5),ObjFunI(5));
%     fprintf(fp,'  6 IRLSc  %1.4f  %1.2f   %6.2f  %.1f\n',   MSEs(6),  Time(6),Err_v(6),ObjFunI(6));
%     fprintf(fp,'-------------------------------------------\n')
%     
    
    
%     [FSC_p2q2, spatialFrequency, meanIntensity] = FourierShellCorrelate(V_p2q2,volref,n,n);
%     [FSC_p2q1, spatialFrequency, meanIntensity] = FourierShellCorrelate(V_p2q1,volref,n,n);
%     [FSC_Mp2q2, spatialFrequency, meanIntensity] = FourierShellCorrelate(V_Mp2q2,volref,n,n);
%     [FSC_Mp2q1, spatialFrequency, meanIntensity] = FourierShellCorrelate(V_Mp2q1,volref,n,n);
%     [FSC_LS, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_LS,volref,n,n);
%     [FSC_LUD, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_LUDADMM,volref,n,n);
%     [FSC_IRLS, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_IRLS,volref,n,n);
    
    % [FSC_LSc, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_LSc,volref,n,n);
    % [FSC_LUDc, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_LUDADMMc,volref,n,n);
    % [FSC_IRLSc, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_IRLSc,volref,n,n);
    % %[FSC_eig, spatialFrequency, meanIntensity] = FourierShellCorrelate(v_eig,volref,n,n);
    
    
%     figure;
%     tick_spacing = 1;
%     plot(FSC_p2q2,'b','LineWidth',2);hold on
%     plot(FSC_Mp2q2,'r--','LineWidth',2);hold on
%     plot(FSC_LS,'g--','LineWidth',2);hold on ;
%     
%     plot(FSC_p2q1,'b--','LineWidth',2);hold on
%     plot(FSC_Mp2q1,'g','LineWidth',2);hold on
%     plot(FSC_LUD,'c','LineWidth',2);hold on ;
%     plot(FSC_IRLS,' k--','LineWidth',2);  hold on ;
    
    % plot(FSC_LSc,'g--','LineWidth',2);hold on ;
    % plot(FSC_LUDc,'c','LineWidth',2);  hold on ;
    % plot(FSC_IRLSc,'r--','LineWidth',2);  hold on ;
    % plot(FSC_eig,'y','LineWidth',2);
    %ylim([0 0.1]); %
%     xlim([1 n/2])
%     ylim([-0.2 1])
    %set(gca,'YTick',[0,0.143,0.5,1])
%     set(gca,'YTick',[-0.2,0,0.2,0.4,0.6, 0.8,1])
%     f_ind_s = num2str([0:0.1:0.5]');
%     set(gca,'XTickLabel',f_ind_s);
%     ylabel('FSC');
%     xlabel('Spatial frequency (1/pixel size)')
%     grid on
%     legend('PGp2q2','PGMp2q2','LS','PGp2q1','PGMp2q1','LUD','IRLS'); set(gca, 'FontSize',18);
%     print('-dpng',strcat('./figsph/vol',fname,num2str(1/SNR),num2str(K),'FSC'));
%     
%     
%     figure;
%     tick_spacing = 1;
%     plot(FSC_p2q2,'b','LineWidth',2);hold on
%     plot(FSC_Mp2q2,'r--','LineWidth',2);hold on
%     plot(FSC_LS,'g--','LineWidth',2);
%     xlim([1 n/2])
%     ylim([-0.2 1])
%     %set(gca,'YTick',[0,0.143,0.5,1])
%     set(gca,'YTick',[-0.2,0,0.2,0.4,0.6, 0.8,1])
%     f_ind_s = num2str([0:0.1:0.5]');
%     set(gca,'XTickLabel',f_ind_s);
%     ylabel('FSC');
%     xlabel('Spatial frequency (1/pixel size)')
%     grid on
%     legend('PGp2q2','PGMp2q2','LS'); set(gca, 'FontSize',18);
%     print('-dpng',strcat('./figsph/vol',fname,num2str(1/SNR),num2str(K),'FSCLS'));
%     
%     
%     figure;
%     tick_spacing = 1;
%     plot(FSC_p2q1,'b--','LineWidth',2);hold on
%     plot(FSC_Mp2q1,'g','LineWidth',2);hold on
%     plot(FSC_LUD,'c','LineWidth',2);hold on ;
%     plot(FSC_IRLS,' k--','LineWidth',2);
%     
%     xlim([1 n/2]) ;
%     ylim([-0.2 1]);
%     %set(gca,'YTick',[0,0.143,0.5,1])
%     set(gca,'YTick',[-0.2,0,0.2,0.4,0.6, 0.8,1])
%     f_ind_s = num2str([0:0.1:0.5]');
%     set(gca,'XTickLabel',f_ind_s);
%     ylabel('FSC');
%     xlabel('Spatial frequency (1/pixel size)')
%     grid on
%     legend('PGp2q1','PGMp2q1','LUD','IRLS'); set(gca, 'FontSize',18);
%     print('-dpng',strcat('./figsph/vol',fname,num2str(1/SNR),num2str(K),'FSCLUD'));
%     
end
end
