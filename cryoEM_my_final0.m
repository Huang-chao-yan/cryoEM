%%%%% please add 'CryoEM_pulin' in your path (HCY 1129)
%%%%% please add WNNM_code in your path for weighted neclear norm (HCY 211212)
%%%%% please add Framelet-TNN-master for estimate the PSNR and SSIM (HCY 211212)
%%%%% QWNNMmotion206025_re+_211212 is for the multi-channel image neclear
%%%%% norm, but it was my perious work, and some parameters are not
%%%%% suitable here, so I add the WNNM_code in the path for some parameters
%%%%% and functions. The svd is not suitable for multi-channel image, I
%%%%% write reshape referenced by the paper of multi-scale WNNM. The PSNR
%%%%% of my WNNM_EM currently is 'Estimated Image: iter = 29.000, PSNR = 84.07'. 
%%%%% It seems convergence. Still need to tune parameters. (HCY 211212)
clear; %close all; %clc;

currentFolder = pwd;
addpath(genpath(currentFolder));


addpath(genpath('CryoEM_pulin'))
addpath(genpath('QWNNM_denoise'))
% addpath(genpath('utilscryoem'))
%rng(10000);
fp = fopen('resultstxt/result0301.txt','a+'); 
% n  = 65; 

    samplepath = ['','smaple0302/s50_1/'];
    samples =dir(samplepath);
    ImgNames = {samples.name}';
    ImgNames = ImgNames(3:end);
    ImgNum = length(ImgNames);
for ii = 1:ImgNum
    fname = ImgNames{ii};
    ph = ReadMRC(fname);  
%ph = cryo_gaussian_phantom_3d('C1_params',n,1); % generate a 3D phantom
%ph = ReadMRC('emd_1252.map');  

% ph= ReadMRC('emd_8719.map');
%https://www.ebi.ac.uk/emdb/EMD-21316?tab=3dview
%https://www.ebi.ac.uk/emdb/EMD-10425?tab=overview

% ph = ph(120:129,30:39,120:129);

%fname = 'emd1252'; 
% fname = 'emd_8719'; 
if mod(size(ph,1),2)==0
    n   = size(ph,1)+1;
    tmp = zeros(n,n,n);
    tmp(1:n-1,1:n-1,1:n-1) = ph;
    ph = tmp;
else
    n = size(ph,1);
end


        for K = [600 1800 5400 10800]                      
        a              = qrand(K);    %
        rotmatrices    = q_to_rot(a); %quat2rotm(a);
        rotmatricesinv = permute(rotmatrices, [2 1 3]);
        %aaa = max(rotmatrices(:)); 
        %rotmatrices    = aaa*0.05*randn(size(rotmatrices)); 

        fprintf(fp,'\n n=%5d, projection NO: %5d\n',n, K); 

        fprintf('************* n=%f            K=%f*****************\n', n,K)


        A     = OpNufft3D(rotmatrices,n); % projection operatorz
        projs = A * ph;      % projected images

        %projections    = cryo_project(ph,a, n,'single'); % generate phantom projecitons
        %projections2   = permute(projections,[2 1 3]);   % transpose each image


        n_r     = ceil(n/2);   %  resolution  of each projection ray  line
        n_theta = 36;          % angle resolution of each projection 

        %%  est_rotation Matrix
        % rot_est = est_rotation(projs,n_r,n_theta,rotmatrices,K); 
        % rot_est_sinv = permute(rot_est, [2 1 3]);
        % fprintf(fp,'estimate rotation matrix \n',rot_est);

        wCTF   = 1; 
        wNoise = 1;

        ctfs      = GenerateCTFset(n);    
        defocusID = mod(0:K-1,size(ctfs,3))+1;
        ctfprojs  = CTFtimesprojs(projs, ctfs, defocusID); % ctfs times projs

        if wCTF && wNoise
            disp('*************************************************')
            disp('**********  with CTF/with noise   ***************')
            disp('*************************************************')
            for SNR = [4 2 1 1/4 1/16 1/64]
                fprintf(fp,'\n\n SNR=%f\n', SNR); 
                fprintf('************* SNR=%f*****************\n', SNR)
            % ctfs      = GenerateCTFset(n);    
            % defocusID = mod(0:K-1,size(ctfs,3))+1;
            % ctfprojs  = CTFtimesprojs(projs, ctfs, defocusID); % ctfs times projs

            [obsimg, sigma] = ProjAddNoise(ctfprojs, SNR); 
            upbound = sum(sigma(:).^2)*n*n;     % eta: the sum of the variance of the noise
            eta0    = sum(obsimg(:).*conj(obsimg(:)));
            erra = norm(obsimg(:)-ctfprojs(:))^2; 
            abs(erra-upbound)/norm(obsimg(:)); 
           % figure; isosurface(real(obsimg),max(real(obsimg(:)))/5);



          %% back_projsections  
           % A     = OpNufft3D(rotmatrices,n);
            Toepker    = compkernel(A,ctfs, defocusID);   %norm(hh(:)-kernel(:)/n/n)/norm(kernel(:))
            ToepCirEig = CryoEMToeplitz2CirEig(Toepker);  % compute kernel eigenvalue of Toeplitz matrix
            ctf_obsimg = CTFtimesprojs(obsimg, ctfs, defocusID); 
            back_projs = A' * ctf_obsimg;  % back projection
            t = 1/SNR; 
            T         = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)); %(A'A + lambda*I)*x
            Treg    = @(x)(CryoEMToeplitzXvol(ToepCirEig,x)+x/t); %(A'A + lambda*I)*x

                %% FIRM method I
%             disp('=======================START==FIRM===============================')
%             fprintf('Reconstruction from clean centered projections affected by CTF\n')
%             [volfirm, v_b, kernel ,err, iter, flag,OutF] = recon3d_firm_ctf(obsimg,ctfs, defocusID, rotmatricesinv,[], 1e-4, 100,zeros(n,n,n),ph,ctf_obsimg); 
%             fprintf('The RE of firm reconstruction is %f.\n',norm(volfirm(:)-ph(:))/norm(ph(:)));
%             fprintf(fp,'The RE of firm reconstruction is %f.\n',norm(volfirm(:)-ph(:))/norm(ph(:)));
%         %     figure; isosurface(real(volfirm),real(max(volfirm(:))/5)); title('wCTF&w Noisy')
%             saldir = './result0301-fbs/';
%             savePath = [saldir fname, '_', num2str(K),'_', num2str(SNR), '_FIRM' '.mat'];
%             save(savePath,'OutF', 'volfirm', 'v_b', 'kernel', 'err', 'iter', 'flag'); 

        %% Tikhonov regularization method
        % Tikhonov Method with fixed aparmeter
        disp('=======================START==Tig===============================')

            t = 1/SNR;      % regularization parameter
            Minvmat = @(x)(x);   %without preconditioner
            [x, errorT, iterT,OutT]  = pcgCryoEM1(Treg,Minvmat, back_projs,  1e-6, 100, back_projs,ph,ctf_obsimg); 
            fprintf('The RE of tikfix reconstruction is %f.\n',norm(x(:)-ph(:))/norm(ph(:)));
            fprintf(fp,'The RE of tikfix reconstruction is %f.\n',norm(x(:)-ph(:))/norm(ph(:)));
        %     figure; isosurface(real(x),max(real(x(:)))/7);
            saldir = './result0301-fbs/';
            savePath = [saldir fname, '_', num2str(K),'_', num2str(SNR), '_Tig' '.mat'];
            save(savePath,'OutT','x', 'errorT', 'iterT'); 

            %tmp   = conj(voltikf) .* (T(voltikf) - 2*back_projs);
            %Jdata = real(sum(tmp(:)) + eta0); 
          %%  Tik_adapt 
            disp('=======================START==Tigad===============================')

             [utika,alpha,OutTAd] = CryoEM_Tik_adapt1(T,back_projs,upbound,eta0,A,ph,ctf_obsimg);
        %     [unew, alpha] = NewtonTikDPRegParam(T,back_projs,upbound,eta0);
            fprintf('The RE of Tikadapt is %f.\n',norm(utika(:)-ph(:))/norm(ph(:)));
            fprintf(fp,'The RE of Tikadapt: %f   \n',norm(utika(:)-ph(:))/norm(ph(:)));
        %     figure; isosurface(real(utika),max(real(utika(:)))/5);
        %     title('wCTF&w Noisy Tikhonov');

            savePath = [saldir fname, '_', num2str(K),'_', num2str(SNR), '_Tigad' '.mat'];
            save(savePath,'OutTAd','utika', 'alpha'); 



        %%  TF FBS
        disp('=======================START==Ours===============================')

            sig= sum(sigma)/length(sigma);
            tic
            OutPutTF = CryoEM_TF_fbs(A,ctf_obsimg,T,back_projs,upbound,eta0,sig,ph);
            OutPuttf=OutPutTF.Sol; 
            toc
             [MPSNRALLpf, SSIMALLpf, FSIMALLpf] = quality(real(OutPuttf)*255, double(ph)*255)
             fprintf('The RE of TF adapt is %f.\n',norm(OutPuttf(:)-ph(:))/norm(ph(:)));
            fprintf(fp,'The PSNR of TF adapt is %f.\n',OutPutTF.psnr);
            fprintf(fp,'The error of number 35 slice is %f.\n', OutPutTF.re);

        %     figure; isosurface(real(OutPuttf),max(real(OutPuttf(:)))/5);
        %     title('wCTF&w Noisy Ours')
             saldir = './result0301-fbs/';
             savePath = [saldir fname, '_', num2str(K),'_', num2str(SNR), '_our' '.mat'];
             save(savePath,'OutPutTF','MPSNRALLpf', 'SSIMALLpf', 'FSIMALLpf'); 





        %%  TV prime dual
        disp('=======================START==TV===============================')

            tic
            OutPutTV = CryoEM_TV_adapt(T,back_projs,upbound,eta0,ph, ctf_obsimg,A);
            OutPuto=OutPutTV.Sol; 
            toc
             [MPSNRALLp, SSIMALLp, FSIMALLp] = quality(real(OutPuto)*255, double(ph)*255)
             fprintf('The RE of TV adapt is %f.\n',norm(OutPuto(:)-ph(:))/norm(ph(:)));
            fprintf(fp,'The RE of TV adapt is %f.\n',norm(OutPuto(:)-ph(:))/norm(ph(:)));
            fprintf(fp,'The error of number 35 slice is %f.\n', OutPutTV.re);

        %     figure; isosurface(real(OutPuto),max(real(OutPuto(:)))/5);
        %     title('wCTF&w Noisy TV')

             saldir = './result0301-fbs/';
             savePath1 = [saldir fname, '_', num2str(K),'_', num2str(SNR), '_TV' '.mat'];
             save(savePath1,'OutPutTV','MPSNRALLp', 'SSIMALLp', 'FSIMALLp'); 

        %     saveresults





            end

            disp('****************  end  *******************')
            %close all; 
        end
        end
        fclose(fp); 
        
end
